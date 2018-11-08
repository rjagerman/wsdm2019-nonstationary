import csv
import re
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

from scipy.sparse import coo_matrix
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

from experiments.util.dataset import ItemDataset
from experiments.util.serialize import save_object_gzipped


def read_tags(file):
    tag_map = defaultdict(list)
    read_tags = {}
    for row in csv.DictReader(file, delimiter="\t"):
        id = int(row['tagID'])
        tags = re.split("[ -]", row['tagValue'].replace("'", ""))
        for tag in tags:
            if tag not in read_tags:
                read_tags[tag] = len(read_tags)
            tag_map[id].append(read_tags[tag])
    return tag_map


def read_artists(file, tag_map):
    artist_tags = defaultdict(list)
    max_tag = 0
    for row in csv.DictReader(file, delimiter="\t"):
        artist_id = int(row['artistID'])
        tag_id = int(row['tagID'])
        tags = tag_map[tag_id]
        max_tag = max(max_tag, max(tags))
        artist_tags[artist_id].extend(tags)

    artist_map = {}
    item_data = []
    item_rows = []
    item_cols = []
    for artist in artist_tags.keys():
        if artist not in artist_map:
            artist_map[artist] = len(artist_map)
        for tag in artist_tags[artist]:
            item_data.append(1)
            item_rows.append(artist_map[artist])
            item_cols.append(tag)
    item_features = coo_matrix((item_data, (item_rows, item_cols)))
    return item_features, artist_map


def read_users(file, artist_map):
    user_map = {}
    user_listens = []
    for row in csv.DictReader(file, delimiter="\t"):
        user_id = int(row['userID'])
        artist_id = int(row['artistID'])
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
            user_listens.append([])
        if artist_id in artist_map:
            user_listens[user_map[user_id]].append(artist_map[artist_id])
    return user_listens, user_map


def read_network(file, user_map):
    adjacency = np.zeros((len(user_map), len(user_map)))
    for row in csv.DictReader(file, delimiter="\t"):
        user_id = int(row['userID'])
        friend = int(row['friendID'])
        adjacency[user_map[user_id], user_map[friend]] = 1.0
        adjacency[user_map[friend], user_map[user_id]] = 1.0
    return adjacency


def main():

    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--datafolder", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--top_clusters", type=int, default=10)
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Construct a map of tag-id -> tag-id for parsed and cleaned tags
    with open(args.datafolder + "/tags.dat", "rt", encoding='latin-1') as file:
        tag_map = read_tags(file)

    # Read artists with tags and convert into feature vectors, then perform PCA
    # to obtain a reduced dimensionality representation
    with open(args.datafolder + "/user_taggedartists.dat",
              encoding='latin-1') as file:
        artist_features, artist_map = read_artists(file, tag_map)
        artist_features = TfidfTransformer(sublinear_tf=True, use_idf=True).fit_transform(artist_features)
        artist_features = TruncatedSVD(25).fit_transform(artist_features)
        id = ItemDataset(artist_features)
        save_object_gzipped(args.datafolder + "/bandit_items.gz", id)

    # Read users and artists
    with open(args.datafolder + "/user_artists.dat",
              encoding='latin-1') as file:
        user_listens, user_map = read_users(file, artist_map)
        id = ItemDataset([np.array(u) for u in user_listens])
        save_object_gzipped(args.datafolder + "/bandit_users.gz", id)

    # Read user graph
    with open(args.datafolder + "/user_friends.dat",
              encoding='latin-1') as file:
        user_clusters = read_network(file, user_map)
        user_clusters = SpectralClustering(n_clusters=args.clusters, gamma=0.1).fit_predict(user_clusters)
        u, c = np.unique(user_clusters, return_counts=True)
        top_clusters = {v: k for k, v in enumerate(u[np.argsort(-c)][0:args.top_clusters])}
        clusters = [[] for _ in range(args.top_clusters)]
        for i in range(user_clusters.shape[0]):
            if user_clusters[i] in top_clusters.keys():
                clusters[top_clusters[user_clusters[i]]].append(i)
        for c in clusters:
            print(f"Cluster size: {len(c)}")
        id = ItemDataset([np.array(c) for c in clusters])
        save_object_gzipped(args.datafolder + "/bandit_clusters.gz", id)


if __name__ == "__main__":
    main()
