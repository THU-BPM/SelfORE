from classifier import Classifier
from sklearn.cluster import KMeans
from adaptive_clustering import AdaptiveClustering
from tqdm import tqdm


class SelfORE:
    def __init__(self, config):
        self.k = config['k']
        self.loop_num = config['loop']
        cluster = config['cluster']
        if cluster == 'kmeans':
            self.pseudo_model = KMeans(n_clusters=self.k, random_state=0)
        elif cluster == 'adpative_clustering':
            self.pseudo_model = AdaptiveClustering(n_clusters=self.k, input_dim=config['bert_max_len']*768)
        else:
            raise Exception(f'Clustering algorithm {cluster} not support yet')

        self.classifier = Classifier(
            k=self.k,
            sentence_path=config['sentence_path'],
            max_len=config['bert_max_len'],
            batch_size=config['batch_size'],
            epoch=config['epoch']
        )

    def loop(self):
        print("=== Generating Pseudo Labels...")
        bert_embs = self.classifier.get_hidden_state()
        pseudo_labels = self.pseudo_model.fit(bert_embs).labels_
        print("=== Generating Pseudo Labels Done")

        print("=== Training Classifier Model...")
        self.classifier.train(pseudo_labels)
        print("=== Training Classifier Model Done")

    def start(self):
        print("starting ...")
        for _ in tqdm(range(self.loop_num)):
            self.loop()
