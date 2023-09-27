import os

from embedders.ml_embedder import MLEmbedder
from stocktake.dataset import StockTakeDataset

from bert_topic import TopicModelRunner
from utils.options import get_args
from config import DATA_FOLDER


def embed():
    """ This function is not used, unless you want to download all source PDFs, run extraction,
        and do pre-processing yourself. In that case, this script should walk you through.
    """
    dataset = StockTakeDataset()
    # Download PDFs.
    dataset.download_pdfs()
    # Extract and pre-process.
    dataset.extract_all_paragraphs()
    dataset.print_paragraph_info()
    # Now run the embedder.
    embedder = MLEmbedder(dataset)
    embedder.run()


if __name__ == '__main__':
    if not os.path.exists(os.path.join(DATA_FOLDER, 'embeddings')):
        raise FileNotFoundError("Embeddings not found. Download our data first using ./utils/download.sh")

    args = get_args()
    topic_model = TopicModelRunner(args)
    # Run or load the topic model.
    if args.load:
        topic_model.load()
    else:
        topic_model.run()
    # Save results to the output folder.
    topic_model.store_results()

    print('YOU ARE TERMINATED!')
