import os
from umap import UMAP
from hdbscan import HDBSCAN
# For non-Windows users, there is also a GPU accelerated version
# from cuml.manifold import UMAP
# from cuml.cluster import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
import numpy as np
from sklearn.decomposition import PCA

import pandas as pd

from embedders.ml_embedder import MLEmbedder
from stocktake.dataset import StockTakeDataset
from config import DATA_FOLDER, OUTPUT_FOLDER, CATEGORIES_KEYWORDS_DICT


__all__ = ['TopicModelRunner']


class TopicModelRunner:
    topic_model = None

    def __init__(self, args):
        self.n_topics = args.num_topics
        self.dataset = StockTakeDataset()
        self.dataset.load_pickled_documents()
        self.embedder = MLEmbedder(self.dataset)
        self.embeddings, self.paragraph_df = self.embedder.load()

        self.paragraphs = self.paragraph_df['paragraph'].to_list()
        self.file_map = pd.read_csv(os.path.join(DATA_FOLDER, 'overview.csv'), encoding='utf-8')

    def run(self):
        """ Run BertTopic on the Stocktake data using the hyperparameters in this function.
            These are the hyperparameters we used, but topic modelling is very sensitive to them, so feel free
            to try different values.
        """
        # 1. Initialize and rescale PCA embeddings.
        # Doing PCA embeddings first helps to reduce compute times. BERTopic documentation suggests using
        # PCA to go down to 5 components, but it seems to work better to keep this number slightly higher.
        pca_embeddings = self._rescale(PCA(n_components=15).fit_transform(self.embeddings))
        dim_model = UMAP(n_neighbors=16, n_components=15, min_dist=0.0, metric='cosine',
                         init=pca_embeddings, random_state=42)
        # 2. BERTopic uses HDBScan to select clusters of similar documents
        # Intuitively, lower nr_topics can allow for higher min_cluster_size.
        # Lowering min_samples then can reduce outliers, but might lead to less "pure" topics.
        hdbscan_model = HDBSCAN(min_cluster_size=30, min_samples=10, metric='euclidean',
                                cluster_selection_method='leaf', prediction_data=True)
        # Use the representation model in BERTopic on top of the default pipeline.
        representation_model = KeyBERTInspired()
        # 3. Run BERTopic.
        self.topic_model = BERTopic(nr_topics=self.n_topics, language='multilingual',
                                    seed_topic_list=list(CATEGORIES_KEYWORDS_DICT.values()),
                                    embedding_model=self.embedder.sentence_transformer,
                                    umap_model=dim_model,
                                    hdbscan_model=hdbscan_model,
                                    representation_model=representation_model,
                                    ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                                    n_gram_range=(1, 2)
                                    )
        print("Running topic modelling.")
        topics, probs = self.topic_model.fit_transform(self.paragraphs, self.embeddings)
        # 4. Reduce outliers
        print("Reducing outliers.")
        new_topics = self.topic_model.reduce_outliers(self.paragraphs, topics, strategy='c-tf-idf', threshold=0.08)
        self.topic_model.update_topics(self.paragraphs, topics=new_topics,
                                       representation_model=representation_model,
                                       ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                                       n_gram_range=(1, 2))
        # Finally, save the model with the default function.
        print("Done. Saving to file.")
        output_file = os.path.join(OUTPUT_FOLDER, f"t{self.n_topics}_model")
        self.topic_model.save(output_file)

    def load(self):
        """ Load from file, if possible.
        """
        output_file = os.path.join(OUTPUT_FOLDER, f"t{self.n_topics}_model")
        if os.path.exists(output_file):
            self.topic_model = BERTopic.load(output_file, embedding_model=self.embedder.sentence_transformer)
        else:
            raise FileNotFoundError(f"{output_file} not found. Try calling run().")

    def store_results(self):
        """ Run a few functions that store and show the results on the trained topic model.
        """
        self._plot_and_save_reduced_embeddings()
        topic_df = self._get_and_save_words_per_topic()
        self._get_and_save_representative_pages_per_topic(topic_df)
        self._get_and_save_topic_distribution(topic_df)

    def _plot_and_save_reduced_embeddings(self):
        pca_embeddings = self._rescale(PCA(n_components=2).fit_transform(self.embeddings))
        reduced_embeddings = UMAP(n_neighbors=16, n_components=2, min_dist=0.0,
                                  metric='cosine', init=pca_embeddings).fit_transform(self.embeddings)
        # Combine this with the article information.
        reduced_df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
        combined_df = pd.concat([self.paragraph_df, reduced_df], axis=1)
        # Add the topic assignment (-1 is an outlier) and save.
        topic_nr_df = pd.DataFrame(self.topic_model.topics_, columns=['topic_nr'])
        combined_df = pd.concat([combined_df, topic_nr_df], axis=1)
        df_path = os.path.join(OUTPUT_FOLDER, f'{self.n_topics}_reduced_embeddings_per_para_report.csv')
        combined_df.to_csv(df_path, encoding='utf-8')
        # Finally, plot.
        fig = self.topic_model.visualize_documents(self.paragraphs, reduced_embeddings=reduced_embeddings,
                                                   hide_document_hover=True, hide_annotations=True)
        fig_path = os.path.join(OUTPUT_FOLDER, f'{self.n_topics}_output_report.html')
        fig.write_html(fig_path)
        fig.show()

    def _get_and_save_words_per_topic(self):
        words_dict = {}
        for label_id, label in self.topic_model.topic_labels_.items():
            if label_id == -1:
                continue
            representative_words = [line[0] for line in self.topic_model.topic_representations_[label_id]]
            words_dict[label] = representative_words
        topic_df = pd.DataFrame.from_dict(words_dict, orient='index')
        df_path = os.path.join(OUTPUT_FOLDER, f'{self.n_topics}_representative_words_report.csv')
        topic_df.to_csv(df_path, encoding='utf-8')
        return topic_df

    def _get_and_save_representative_pages_per_topic(self, topic_df):
        del self.topic_model.representative_docs_[-1]  # outliers
        rdocs_df = pd.DataFrame.from_dict(self.topic_model.representative_docs_, orient='index',
                                          columns=['Most_representative_paragraph1', 'Most_representative_paragraph2',
                                                   'Most_representative_paragraph3'])
        combined_df = pd.concat([topic_df.reset_index(), rdocs_df], axis=1)
        df_path = os.path.join(OUTPUT_FOLDER, f'{self.n_topics}_representative_docs_per_topic_report.csv')
        combined_df.to_csv(df_path, encoding='utf-8')

    def _get_and_save_topic_distribution(self, topic_df):
        topic_distr, _ = self.topic_model.approximate_distribution(self.paragraphs, min_similarity=0.08,
                                                              window=5, stride=2,
                                                              use_embedding_model=True, batch_size=50000)
        distr_df = pd.DataFrame(topic_distr, columns=topic_df.index)
        df = pd.concat([self.paragraph_df, distr_df], axis=1)
        df_path = os.path.join(OUTPUT_FOLDER, f'{self.n_topics}_theta_embeddings_report.csv')
        df.to_csv(df_path, encoding='utf-8')

    @staticmethod
    def _rescale(x, inplace=False):
        """ Rescale an embedding so optimization will not have convergence issues.
        """
        if not inplace:
            x = np.array(x, copy=True)
        x /= np.std(x[:, 0]) * 10000
        return x
