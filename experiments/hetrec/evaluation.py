from os import path

import numpy as np
import json
from argparse import ArgumentParser

from chainer import as_variable
from chainercb.policies import EpsilonGreedy

from experiments.estimators import IPS, DMADF, AverageAggregator, DRADF, \
    ExponentialAggregator, WindowAggregator, AdaptiveExponentialAggregator, \
    AdaptiveWindowAggregator
from experiments.util.hetrec import InstanceGenerator, LimitedIterator
from experiments.util.serialize import load_object_gzipped
from experiments.util.time_shift import DictShift, IncBetaShift, \
    TransformShift, DurationShift, CompositeShift, LinearShift, AbruptShift


def write_results(results, args):
    with open(args.output + '/log.json', 'wt') as f:
        json.dump(results, f, indent=2)


def reward(actions):
    correct = np.zeros_like(actions.data, dtype=actions.data.dtype)
    return as_variable(1.0 * (correct == actions.data))


def evaluate_true_reward(it, policy):
    it.reset()
    n = 0
    sum_reward = 0.0
    for batch in it:
        batch = np.stack(batch)
        n += len(batch)
        a = policy.max(batch)
        r = reward(a)
        sum_reward += np.sum(r.data)
    return sum_reward / n


def main():

    estimator = {
        'ips': lambda policy: IPS(policy),
        'dm': lambda policy: DMADF(policy, 25),
        'dr': lambda policy: DRADF(policy, 25)
    }

    aggregator = {
        'exp': lambda est, args: ExponentialAggregator(est, args.alpha),
        'avg': lambda est, args: AverageAggregator(est),
        'win': lambda est, args: WindowAggregator(est, args.tau),
        'adaexp': lambda est, args: AdaptiveExponentialAggregator(est,
                                                                  args.alpha,
                                                                  factor=args.factor,
                                                                  adapt_window=args.adapt),
        'adawin': lambda est, args: AdaptiveWindowAggregator(est, args.tau,
                                                             factor=args.factor,
                                                             adapt_window=args.adapt)
    }

    change = {
        'linear': lambda: LinearShift(),
        'smooth': lambda: IncBetaShift(2.5),
        'abrupt': lambda: AbruptShift(),
        'stationary': lambda: None,
    }

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--logging", type=str, required=True)
    parser.add_argument("--policies", nargs="+", type=str, required=True)
    parser.add_argument("--estimator", choices=estimator.keys(), required=True)
    parser.add_argument("--aggregator", choices=aggregator.keys(), required=True)
    parser.add_argument("--change", choices=change.keys(), required=True)
    parser.add_argument("--datafolder", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--output_every", type=int, default=100)
    parser.add_argument("--iter", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test_iter", type=int, default=20000)
    parser.add_argument("--test_batch_size", type=int, default=5000)
    parser.add_argument("--test_every", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.9999)
    parser.add_argument("--alpha2", type=float, default=0.999)
    parser.add_argument("--adapt", type=int, default=10000)
    parser.add_argument("--tau", type=int, default=200)
    parser.add_argument("--tau2", type=int, default=200)
    parser.add_argument("--factor", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--truek", action='store_true')
    parser.add_argument("--skip_true_eval", action='store_true')
    parser.add_argument("--store_batches", type=str, default=None)
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Save parameters
    dictionary = vars(args)
    with open(path.join(args.output, 'args.json'), 'wt') as f:
        json.dump(dictionary, f, indent=2)

    # Load data sets
    users = load_object_gzipped(args.datafolder + "/bandit_users.gz")
    artists = load_object_gzipped(args.datafolder + "/bandit_items.gz")
    clusters = load_object_gzipped(args.datafolder + "/bandit_clusters.gz")

    # Construct an instance generator over our data set
    ig = InstanceGenerator(args.batch_size, users, artists, clusters)
    ig2 = InstanceGenerator(args.test_batch_size, users, artists, clusters)
    ig2.cluster_mixture = ig.cluster_mixture

    # Create limited instance generator for iterating over the dataset
    it = LimitedIterator(ig, args.iter)
    test_it = LimitedIterator(ig2, args.test_iter)

    # Load logging policy
    logger = EpsilonGreedy(load_object_gzipped(args.logging), 0.2)
    policies = [load_object_gzipped(p) for p in args.policies]
    estimators = [estimator[args.estimator](p) for p in policies]
    aggregators = [aggregator[args.aggregator](estimator, args)
                  for estimator in estimators]

    # Non-stationarity
    shifts = []
    if args.change != "stationary":
        shifts = [DictShift(ig.cluster_mixture, p, CompositeShift([], 0.0))
                  for p in range(len(policies))]
        shift_len = args.iter / len(policies)
        for p in range(1, len(policies)):

            # p-1 should be going down
            shift_down = TransformShift(1.0, 0.0, change[args.change]())
            d_down = DurationShift((p - 1) * shift_len, p * shift_len, shift_down)
            shifts[p - 1].shift_fn.append(d_down)

            # p should go up
            shift_up = TransformShift(0.0, 1.0, change[args.change]())
            d_up = DurationShift((p - 1) * shift_len, p * shift_len, shift_up)
            shifts[p].shift_fn.append(d_up)

    # Iterate over data
    n = 0
    results = []
    true_rewards = [None for _ in range(len(policies))]
    for t, batch in enumerate(it):

        # Perform non-stationary shift of the cluster distributions
        for shift in shifts:
            shift(n)
        result = {"iteration": n}

        # Get batch and compute actions, propensities and rewards
        batch = as_variable(np.stack(batch))
        n += batch.shape[0]
        a = logger.draw(batch)
        p = logger.propensity(batch, a)
        r = reward(a)

        # Update estimators
        for i, policy in enumerate(policies):
            result[f"policy_{i}/estimate"] = aggregators[i](batch, a, p, r)

        # Compute an MC estimate of the true reward when needed and display the
        # estimates vs the true rewards
        if not args.skip_true_eval and (t % args.test_every == 0):
            true_reward = evaluate_true_reward(test_it, logger)
            result[f"logging_policy/true"] = true_reward
            for i in range(len(policies)):
                prev_true_reward = true_rewards[i]
                result[f"policy_{i}/true"] = evaluate_true_reward(test_it,
                                                                  policies[i])
                true_rewards[i] = result[f"policy_{i}/true"]
                diff = result[f"policy_{i}/true"] - result[f"policy_{i}/estimate"]
                result[f"policy_{i}/mse"] = diff ** 2
                result[f"policy_{i}/mae"] = abs(diff)

                true_k = 0.0
                if prev_true_reward is not None:
                    true_k = (true_rewards[i] - prev_true_reward) / (args.test_every * args.batch_size)

                result[f"policy_{i}/true_k"] = true_k
                if args.aggregator in ["adaexp", "adawin"]:
                    result[f"policy_{i}/est_k"] = aggregators[i].k

                if args.truek and prev_true_reward is not None:
                    print(abs(prev_true_reward - true_rewards[i]))
                    aggregators[i].true_k = abs(prev_true_reward - true_rewards[i]) / (args.test_every * args.batch_size)

        # Write results
        results.append(result)
        if t % args.output_every == 0:
            write_results(results, args)

        if (t * args.batch_size) % 1000 == 0:
            print(f"Progress: {t * args.batch_size} / {args.iter}")

    # Write results one last time
    write_results(results, args)


if __name__ == "__main__":
    main()
