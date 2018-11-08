from time import time

import numpy as np
from chainer.dataset import Iterator as _Iterator

from experiments.util.dataset import ItemDataset


class InstanceGenerator(_Iterator):
    """
    A generator which generates evaluation instances according to a generative
    process.
    """
    def __init__(self, batch_size, user_dataset, artist_dataset,
                 clusters_dataset, candidate_set_size=25):
        """
        :param batch_size: The size of each batch to generate
        :type batch_size: int

        :param user_dataset: The dataset of listened artists per user
        :type user_dataset: experiments.util.dataset.ItemDataset

        :param artist_dataset: The dataset of feature vectors per artist
        :type artist_dataset: experiments.util.dataset.ItemDataset

        :param clusters_dataset: The dataset of users per cluster
        :type cluster_dataset: experiments.util.dataset.ItemDataset

        :param candidate_set_size: The size of the candidate artists in each
                                   generated instance.
        :type candidate_set_size: int
        """
        self.batch_size = batch_size

        self.user_dataset = user_dataset
        self.artist_dataset = artist_dataset
        self.clusters_dataset = clusters_dataset

        self.nr_users = len(self.user_dataset)
        self.nr_artists = len(self.artist_dataset)
        self.nr_clusters = len(self.clusters_dataset)
        self.out_size = candidate_set_size

        self.clusters_weights = ItemDataset([np.ones(len(x)) / len(x)
                                             for x in self.clusters_dataset])
        self.cluster_sizes = np.array([len(x) for x in self.clusters_dataset])
        self.cluster_mixture = np.ones(self.nr_clusters) / self.nr_clusters

        # Data structures for sampling
        self.user_artist_reward = np.zeros((len(self.user_dataset),
                                            len(self.artist_dataset)),
                                           dtype=np.int32)
        for i, u in enumerate(self.user_dataset):
            self.user_artist_reward[i, u] = 1
        self.sorted_reward_idx = np.argsort(-self.user_artist_reward, axis=1)
        self.sorted_reward_loc = np.sum(self.user_artist_reward == 1, axis=1)
        self.sorted_reward_idx = self.sorted_reward_idx.flatten()

    def __next__(self):
        # Sample from clusters proportionally
        cs = np.random.choice(self.nr_clusters, size=self.batch_size,
                              p=self.cluster_mixture)
        cs = np.hstack(cs)

        # Account for different cluster sizes via re-weighting
        w = np.hstack(self.clusters_weights[cs])
        w /= np.sum(w)

        # Select users from the clusters proportionally
        us = np.random.choice(np.hstack(self.clusters_dataset[cs]),
                              size=self.batch_size, p=w)

        # Sample instances
        final_idx = self._sample_per_item(us)

        # Look up feature vectors for selected artists and reshape back into
        # samples
        result = np.vstack(self.artist_dataset[final_idx])
        result = result.reshape((self.batch_size, self.out_size,
                                 result.shape[1]))
        return result

    def _sample_per_item(self, us):

        # We use flattened indices, this will correct for user locations
        us_idx = us * self.nr_artists

        # Samples using Floyd algorithm, as described in
        # https://stackoverflow.com/questions/2394246/algorithm-to-select-a-single-random-combination-of-values
        # This implementation is more efficient than shuffling than grabbing
        # first out_size entries because nr_artists >> out_size
        out = -np.ones((us.shape[0], self.out_size), dtype=np.int32)
        n = self.nr_artists - self.sorted_reward_loc[us]
        r = np.random.randint(2 ** 31, size=(us.shape[0], self.out_size))

        for i in range(self.out_size - 1, 0, -1):
            j = n - i
            r_j = np.mod(r[:, i], j)
            s = 1 * (np.sum(out == r_j[:, None], axis=1) > 0)
            v = s * j + (1 - s) * r_j
            out[:, i] = v

        out[:, 0] = np.mod(r[:, 0], self.sorted_reward_loc[us])
        out[:, 1:] += self.sorted_reward_loc[us][:, None]
        out += us_idx[:, None]
        out = self.sorted_reward_idx[out.flatten()]
        return out

    # def _sample_slow(self, us):
    #     # For the users, get the reward per artist. Next, we perform an argsort
    #     # so that positive items are at the end and negative in the beginning
    #     # the random part in lexsort ensures that tie breaks happen randomly
    #     r = self.user_artist_reward[us, :]
    #     idx = np.lexsort((np.random.random(r.shape), r), axis=1)
    #
    #     # Get positive and negative samples
    #     pos_idx = idx[:, -2:-1]
    #     neg_idx = idx[:, 0:24]
    #
    #     # Get artist indices as a single flattened array
    #     return np.hstack((pos_idx, neg_idx)).flatten()
    #
    # def _sample_fast(self, us):
    #     # We use flattened indices to do efficient lookup of different users and
    #     # their clicked / not-clicked articles, this array can be used to
    #     # correct the position indices in the flattened user_artist matrix
    #     pos_u_idx = us * self.nr_artists
    #
    #     # Compute indices for positive items for each selected user
    #     pos_idx = np.random.randint(2 ** 31, size=us.shape[0])
    #     pos_idx = np.mod(pos_idx, self.sorted_reward_loc[us])
    #
    #     # Get artist indices for the positive items
    #     pos_art = self.sorted_reward_idx[pos_u_idx + pos_idx]
    #
    #     # Compute indices for negative items for each selected user. The if
    #     # statement is an optimization where the way we sample items is drawn
    #     # from a different sized set depending on what the batch size is
    #     if self.batch_size > 10:
    #         neg_idx = self._neg_large(us)
    #     else:
    #         neg_idx = self._neg_small(us)
    #     neg_idx = neg_idx + self.sorted_reward_loc[us][:, None]
    #     neg_idx = neg_idx.flatten()
    #
    #     # Similar to pos_u_idx, this is to take care of correcting for position
    #     # indices in the flattened neg_user_artist matrix
    #     neg_u_idx = np.repeat(pos_u_idx, self.out_size - 1)
    #
    #     # Get the negative items for the selected users
    #     neg_art = self.sorted_reward_idx[neg_u_idx + neg_idx]
    #
    #     # Reshape back to matrix form for slicing (1 positive with c negative)
    #     # and then convert back to flattened array form
    #     pos_art = pos_art[:, None]
    #     neg_art = neg_art.reshape((self.batch_size, self.out_size - 1))
    #
    #     # Get artist indices as a single flattened array
    #     return np.hstack([pos_art, neg_art]).flatten()
    #
    # def _neg_large(self, us):
    #     neg_idx = np.random.choice(2 ** 20, replace=False,
    #                                size=(self.batch_size, self.out_size - 1))
    #     neg_mod = self.nr_artists - self.sorted_reward_loc[us]
    #     return np.mod(neg_idx, neg_mod[:, None])
    #
    # def _neg_small(self, us):
    #     neg_idx = np.random.choice(2 ** 16, replace=False,
    #                                size=(self.batch_size, self.out_size - 1))
    #     neg_mod = self.nr_artists - self.sorted_reward_loc[us]
    #     return np.mod(neg_idx, neg_mod[:, None])


class LimitedIterator(_Iterator):
    """
    Iterates over batches from given iterator for a limited number of samples
    """
    def __init__(self, iterator, limit):
        """
        :param iterator: The iterator to iterate over
        :type iterator: chainer.dataset.Iterator

        :param limit: The maximum number of samples
        :type limit: int
        """
        self.limit = limit
        self.iterator = iterator
        self.count = 0

    def reset(self):
        """
        Resets the iterator so it can iterate again.
        """
        self.count = 0

    def __next__(self):
        if self.count < self.limit:
            batch = self.iterator.__next__()
            samples_left = self.limit - self.count
            if len(batch) > samples_left:
                batch = batch[0:samples_left]
            self.count += len(batch)
        else:
            raise StopIteration
        return batch


def _index_per_row(x, i):
    idx = i.flatten() + np.repeat(np.arange(i.shape[0]) * i.shape[1], i.shape[1])
    return x.flatten()[idx].reshape(x.shape)
