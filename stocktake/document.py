import os
import re
import pandas
import pickle

from config import DATA_FOLDER
from stocktake.text_extractor import extract_all_text_from_pdfs_using_tesseract

RELEVANT_COLUMNS = ['author', 'author_is_party', 'date', 'document_id', 'document_md5_sum', 'document_name',
                    'document_family_id', 'document_source_url', 'document_variant', 'language', 'types', 'version']


class Document:
    path, paragraphs, language = None, None, None

    def extract_text(self):
        self.paragraphs, self.language = extract_all_text_from_pdfs_using_tesseract(self.path)

    def __getstate__(self):
        return {key: getattr(self, key) for key in self.__dict__}

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])

    def to_pickle(self):
        if not self.paragraphs:
            return
        pickle_path = re.sub('pdfs', 'docs', self.path)
        pickle_path = re.sub('.pdf', '.pkl', pickle_path)
        if not os.path.exists(os.path.join(DATA_FOLDER, 'docs')):
            os.mkdir(os.path.join(DATA_FOLDER, 'docs'))
        with open(pickle_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def from_pickle(md5_sum):
        pickle_path = os.path.join(DATA_FOLDER, 'docs', f'{md5_sum}.pkl')
        if not os.path.exists(pickle_path):
            return None
        with open(pickle_path, 'rb') as pickle_file:
            p = pickle.load(pickle_file)
            # The pickle includes the path to the file, but this is based on the machine that made the pickle => swap.
            p.path = os.path.join(DATA_FOLDER, 'pdfs', os.path.split(p.path)[-1])
        return p


class StockTakeDocument(Document):
    doc_id, name, md5_sum, source_url, family_id, variant = None, None, None, None, None, None
    author, author_is_party, date, language, types, version = None, None, None, None, None, None

    def set_from_textdf_series(self, line):
        relevant_data = line[RELEVANT_COLUMNS]
        for col, value in relevant_data.items():
            setattr(self, col, value)
        self.doc_id = relevant_data.name
        assert self.types[0] == '[' and self.types[-1] == ']', "Not using eval() on something that is no list."
        self.types = eval(self.types)
        assert self.author[0] == '[' and self.author[-1] == ']', "Not using eval() on something that is no list."
        self.author = eval(self.author)
        self.path = os.path.join(DATA_FOLDER, 'pdfs', f'{self.md5_sum}.pdf')

    def set_from_overview_series(self, line):
        for col, value in line.items():
            setattr(self, col, value)
        self.author = self.author.split(';')
        self.types = self.types.split(';')
        self.path = os.path.join(DATA_FOLDER, 'pdfs', f'{self.md5_sum}.pdf')

    def get_pandas_series(self):
        attribute_dict = {var: getattr(self, var) for var in vars(self)}
        del attribute_dict['path']
        del attribute_dict['paragraphs']
        attribute_dict['author'] = ';'.join(attribute_dict['author'])
        attribute_dict['types'] = ';'.join(attribute_dict['types'])
        return pandas.Series(attribute_dict)

    def __str__(self):
        return f'Document\n{self.author}\t{self.date}\t{self.name}\n' \
               f'Checksum:\t{self.md5_sum}\n' \
               f'URL:\t{self.source_url}'
