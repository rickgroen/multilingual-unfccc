import pickle

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from config import DATA_FOLDER, EMBEDDER


class MLEmbedder(object):

    def __init__(self, dataset):
        self.which_sentence_transformer = EMBEDDER
        self.sentence_transformer = SentenceTransformer(self.which_sentence_transformer)
        self.dataset = dataset
        # Make a dir in which we put embeddings of processed primitives.
        self.embeddings_dir = os.path.join(DATA_FOLDER, 'embeddings')
        if not os.path.exists(self.embeddings_dir):
            os.mkdir(self.embeddings_dir)

    def _print_paragraph_overview(self):
        self.dataset.print_paragraph_info()

    def load(self):
        """ Returns a list of embeddings and a pandas df with the texts per paragraph + identifiers
            This will take a fair bit of memory! ~10 GB for our dataset
        """
        assert os.path.exists(self.embeddings_dir)
        assert len([d for d in self.dataset.documents if d is None]) == 0, "There should be no empty documents."
        print(f"Loading embeddings from {self.embeddings_dir}")
        embedding_list, paragraph_list, md5_list, para_ids, language_list = [], [], [], [], []
        # The embeddings_dir has pickles per document (which can have multiple paragraphs)
        # load the pickle for each document, then add each item from this dict
        for document in tqdm(self.dataset.documents):
            pickle_path = os.path.join(self.embeddings_dir, f'{document.md5_sum}.pkl')
            with open(pickle_path, 'rb') as pickle_file:
                embeddings_dict = pickle.load(pickle_file)
            assert len(embeddings_dict) == len(document.paragraphs)
            for paragraph_id, embedding in embeddings_dict.items():
                embedding_list.append(embedding)
                paragraph_list.append(document.paragraphs[paragraph_id])
                md5_list.append(document.md5_sum)
                para_ids.append(paragraph_id)
                language_list.append(document.language)
        # Make into a dataframe.
        df = pd.DataFrame({'document_md5': md5_list,
                           'paragraph_id': para_ids,
                           'paragraph': paragraph_list,
                           'language': language_list}
                          )
        return np.asarray(embedding_list), df

    def run(self):
        print(f"Running embedder on {self.dataset.print_paragraph_info()}")
        # We run documents sequentially. Not the quickest... but easy to follow
        for document in tqdm(self.dataset.documents):
            # Embed all paragraphs and then save them to a dict.
            embeddings = self.sentence_transformer.encode(document.paragraphs)
            embedding_dict = {idx: embeddings[idx] for idx in range(len(document.paragraphs))}
            # Finally, save the embedding dict as a .pkl file to
            save_path = os.path.join(self.embeddings_dir, f'{document.md5_sum}.pkl')
            with open(save_path, 'wb') as embedding_file:
                pickle.dump(embedding_dict, embedding_file)
