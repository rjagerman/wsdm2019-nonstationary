from chainer.dataset import DatasetMixin as _DatasetMixin


class ItemDataset(_DatasetMixin):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def get_example(self, i):
        return self.items[i]
