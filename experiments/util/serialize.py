import gzip
import pickle


def save_object_gzipped(path, target, mode='wb'):
    with gzip.open(path, mode) as f:
        pickle.dump(target, f, protocol=4)


def load_object_gzipped(path, mode='rb'):
    with gzip.open(path, mode) as f:
        return pickle.load(f)
