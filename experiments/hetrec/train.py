import numpy as np

from experiments.util.hetrec import InstanceGenerator, LimitedIterator
from argparse import ArgumentParser
from chainer import as_variable
from chainercb.policies import ADFUCBPolicy

from experiments.util.serialize import load_object_gzipped, save_object_gzipped


def main():
    parser = ArgumentParser()
    parser.add_argument("--datafolder", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--specialized", type=int, default=None, required=False)
    parser.add_argument("--save", type=str, default=None, required=False)
    args = parser.parse_args()

    # Load data sets
    np.random.seed(args.seed)
    users = load_object_gzipped(args.datafolder + "/bandit_users.gz")
    artists = load_object_gzipped(args.datafolder + "/bandit_items.gz")
    clusters = load_object_gzipped(args.datafolder + "/bandit_clusters.gz")

    # Construct an iterator over our data set
    ig = InstanceGenerator(args.batch_size, users, artists, clusters)
    if args.specialized is not None and 0 <= args.specialized < ig.cluster_mixture.shape[0]:
        ig.cluster_mixture[:] = 0.0
        ig.cluster_mixture[args.specialized] = 1.0
    lig = LimitedIterator(ig, args.iterations)

    # Train a model
    adfucb = ADFUCBPolicy(25)
    cumulative_reward = 0.0
    current_reward = 0.0
    n = 0
    i = 0
    n_c = 0
    for batch in lig:
        n += len(batch)
        n_c += len(batch)
        i += 1
        x = as_variable(np.stack(batch))
        a = adfucb.draw(x)
        z = np.zeros_like(a)
        r = as_variable(1.0 * (a.data == z))
        adfucb.update(x, a, None, r)
        current_reward += np.sum(r.data)
        cumulative_reward += np.sum(r.data)
        if n % 1000 == 0:
            print(f"{n:6d}    {n - int(cumulative_reward):7d}     {cumulative_reward / n:5f}    {current_reward / n_c:5f}")
            current_reward = 0.0
            n_c = 0

    # Save the model
    if args.save is not None:
        save_object_gzipped(args.save, adfucb)


if __name__ == "__main__":
    main()
