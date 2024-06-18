import os
import shutil
import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN


def check_fix_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def check_del_path(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


class IPv62VecCluster:
    def __init__(self, source_data: str, ipv62vec_profile: str, out_path: str):
        self.__source_data = source_data
        self.__ipv62vec_profile = ipv62vec_profile
        self.__out_path = out_path

    def create_category(self):
        f = open(self.__source_data, 'r')
        raw_data = f.readlines()
        f.close()
        index_alpha = '0123456789abcdefghijklmnopqrstuv'
        address_sentences = []
        for address in raw_data:
            address = address.strip().replace(':', '')
            address_words = []
            for nybble, index in zip(address[:-1], index_alpha):
                address_words.append(nybble + index)
            address_sentences.append(' '.join(address_words) + '\n')

        f = open(self.__ipv62vec_profile, 'w')
        f.writelines(address_sentences)
        f.close()

        sentences = word2vec.LineSentence(self.__ipv62vec_profile)
        model = word2vec.Word2Vec(sentences, alpha=0.025, min_count=0, vector_size=100, window=5,
                                  sg=0, hs=0, negative=5, ns_exponent=0.75, epochs=5)

        vocab = list(model.wv.index_to_key)
        X_tsne = TSNE(n_components=2, learning_rate=200, perplexity=30).fit_transform(model.wv[vocab])

        address_split_sentences = [sentence[:-1].split(' ') for sentence in address_sentences]
        x = []
        y = []
        for address_split_sentence in address_split_sentences:
            x_one_sample = []
            y_one_sample = []
            for word in address_split_sentence:
                x_one_sample.append(X_tsne[vocab.index(word), 0])
                y_one_sample.append(X_tsne[vocab.index(word), 1])
            x.append(np.mean(x_one_sample))
            y.append(np.mean(y_one_sample))

        dim_reduced_data = []
        for i, j in zip(x, y):
            dim_reduced_data.append([i, j])
        dim_reduced_data = pd.DataFrame(dim_reduced_data)
        dim_reduced_data.columns = ['x', 'y']

        data = self.cluster(dim_reduced_data)
        self.search_cluster(data, raw_data)

    def cluster(self, data):
        db = DBSCAN(eps=0.0085, min_samples=64).fit(data)
        data['labels'] = db.labels_
        n_clusters_ = 0
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        if n_clusters_ > 10:
            raise ValueError(
                'ClassNumError: generated class num %s > 10, please reset IPv62Vec parameters.' % n_clusters_)
        print('cluster num', n_clusters_)
        return data

    def search_cluster(self, data, raw_data):
        index_list = []
        labels = set(data['labels'])
        for label in labels:
            index_list.append(data[data['labels'] == label].index)

        for index, label in zip(index_list, labels):
            out_path = os.path.join(self.__out_path, 'cluster_' + str(label) + '.txt')
            f = open(out_path, 'w')
            for i in index:
                f.write(raw_data[i])
            f.close()
giy