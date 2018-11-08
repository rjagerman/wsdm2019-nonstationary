from chainer import Chain, functions as F, report, as_variable
from chainer.backends import cuda
from chainer.iterators import MultithreadIterator
from chainer.training.extensions import Evaluator
from chainerltr.dataset import load
from chainerltr.evaluation import ndcg

from experiments.util.concat import ZeropadAsync
from experiments.util.trigger import LogscaleTrigger


def add_evaluator_if_needed(name, evaluation_chain, trainer, args,
                            name_suffix=''):
    dataset_location = vars(args).get(name)
    if dataset_location is not None:
        data = load(dataset_location, cache_path=args.cache, normalize=True,
                    filter=True)
        data_iterator = MultithreadIterator(data,
                                            batch_size=args.eval_batch_size,
                                            shuffle=False, repeat=False,
                                            n_threads=2)
        evaluator = Evaluator(data_iterator, evaluation_chain,
                              converter=ZeropadAsync(), device=args.device)
        if args.only_eval_end:
            evaluator.trigger = (args.iterations, 'iteration')
        else:
            evaluator.trigger = LogscaleTrigger()
        if name_suffix != '':
            name = f'{name}/{name_suffix}'
        trainer.extend(evaluator, name=name)


class NdcgEvaluator(Chain):
    def __init__(self, ranker):
        super().__init__(ranker=ranker)

    def __call__(self, x, y, nr_docs):
        x, y, nr_docs = as_variable(x), as_variable(y), as_variable(nr_docs)
        det_a = self.ranker.max(x)
        sto_a = self.ranker.draw(x)
        self.report_scores(det_a, y, nr_docs)
        self.report_scores(sto_a, y, nr_docs, prefix='stochastic/')
        return as_variable(y)

    def report_scores(self, a, y, nr_docs, prefix=''):
        score = ndcg(a, y, nr_docs, k=-1)
        for i in range(1, 11):
            index = min(i, score.shape[1] - 1)
            report({f"{prefix}ndcg@{i}": F.mean(score[:, index])}, self)
        report({f"{prefix}ndcg": F.mean(score)}, self)


class CTREvaluator(Chain):
    def __init__(self, ranker, click_model):
        super().__init__(ranker=ranker, click_model=click_model)

    def __call__(self, x, y, nr_docs):
        x, y, nr_docs = as_variable(x), as_variable(y), as_variable(nr_docs)
        det_a = self.ranker.max(x)
        sto_a = self.ranker.draw(x)
        self.report_scores(det_a, y, nr_docs, x.dtype)
        self.report_scores(sto_a, y, nr_docs, x.dtype, prefix='stochastic/')
        return as_variable(y)

    def report_scores(self, a, y, nr_docs, dtype, prefix=''):
        cv = self.click_model(a, y, nr_docs)
        cv = as_variable(cv.data.astype(dtype))

        xp = cuda.get_array_module(a)
        """:type : numpy"""

        one_to_n = xp.arange(0.0, cv.shape[1], 1.0, dtype=cv.dtype) + 1.0
        one_to_n = one_to_n.reshape((1, one_to_n.shape[0]))
        one_to_n = xp.broadcast_to(one_to_n, cv.shape)
        ctr = F.cumsum(cv, axis=1) / one_to_n

        for i in range(0, 10):
            index = min(i, ctr.shape[1] - 1)
            report({f"{prefix}ctr@{i + 1}": F.mean(ctr[:, index])}, self)
        report({f"{prefix}ctr": F.mean(ctr[:, -1])}, self)
