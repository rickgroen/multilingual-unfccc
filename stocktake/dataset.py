import os
import time
import pandas
from tqdm import tqdm

from stocktake.document import StockTakeDocument
from utils.webutils import download
from config import DATA_FOLDER


class StockTakeDataset:
    overview_file = os.path.join(DATA_FOLDER, 'overview.csv')
    documents = []

    def __init__(self):
        if not os.path.exists(self.overview_file):
            raise FileNotFoundError("Please download our data first using ./utils/download.sh")
        self.__init_from_overview()

    def __init_from_overview(self):
        document_dataframe = pandas.read_csv(self.overview_file)
        for idx, row in document_dataframe.iterrows():
            doc = StockTakeDocument()
            doc.set_from_overview_series(row)
            self.documents.append(doc)

    def load_pickled_documents(self):
        for idx, document in enumerate(self.documents):
            self.documents[idx] = StockTakeDocument().from_pickle(document.md5_sum)
        self.documents = [document for document in self.documents if document]

    def __save_overview(self):
        docs_to_series = [document.get_pandas_series() for document in self.documents]
        document_dataframe = pandas.DataFrame(docs_to_series)
        document_dataframe.to_csv(self.overview_file, index=False)

    def download_pdfs(self):
        pdf_dir = os.path.join(DATA_FOLDER, 'pdfs')
        if not os.path.exists(pdf_dir):
            os.mkdir(pdf_dir)
        number_of_downloaded_pdfs = 0
        for document in self.documents:
            download_link = document.source_url
            filename = f'{document.md5_sum}.pdf'
            success = download(download_link, pdf_dir, filename)
            time.sleep(2)
            number_of_downloaded_pdfs += int(success)
        print(f'Retrieved {number_of_downloaded_pdfs} PDFs.')

    def extract_all_paragraphs(self):
        for document in tqdm(self.documents):
            document.extract_text()
            document.to_pickle()
        self.documents = [document for document in self.documents if document.paragraphs]
        self.print_paragraph_info()

    def print_paragraph_info(self):
        number_of_paragraphs = sum([len(doc.paragraphs) for doc in self.documents])
        sum_of_words = sum([len(paragraph.split()) for doc in self.documents for paragraph in doc.paragraphs])
        print(f'Extracted {number_of_paragraphs} paragraphs with an average of {sum_of_words / number_of_paragraphs} '
              f'words per paragraph ({sum_of_words}).')
        return number_of_paragraphs

    def print_duplicates(self):
        md5_sums = [doc.md5_sum for doc in self.documents]
        duplicates = [(idx, line) for idx, line in enumerate(md5_sums) if md5_sums.count(line) > 1]
        print(duplicates)

    def filter_by_name(self, filter_words=['archived', 'cover letter', 'table', 'experts']):
        self.documents = [doc for doc in self.documents if not any([word in doc.name.lower() for word in filter_words])]
        self.__save_overview()

    def __str__(self):
        return f'Stocktake Dataset ({len(self)} Documents).'

    def __len__(self):
        return len(self.documents)
