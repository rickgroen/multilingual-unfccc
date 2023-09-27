import os

# Determine which folder to store your data in.
DATA_FOLDER = '/home/data/stocktake/'  # <== PUT YOUR DATA FOLDER HERE.
OUTPUT_FOLDER = os.path.join('.', 'output')

# Some text pre-processing magic variables.
SENTENCE_MIN_WORDS = 8
SENTENCE_MIN_CHARACTERS = 32
SENTENCE_MAX_WORDS = 256
MIN_SENTENCES_PER_PARAGRAPH = 2
SPLIT_PARAGRAPH_WORDS = 100

# Set the type of embedder to use
EMBEDDER = 'sentence-transformers/distiluse-base-multilingual-cased-v1'

# A list of guided topic used for running a guided topic model.
CATEGORIES_KEYWORDS_DICT = {
    "Key 1": ["Paris Agreement", "climate action", " goal",
              ],
    "Key 2": ["sustainable development", "SDG", "eradicate poverty", "mainstream",],
    "Key 3": ["systems transformations", " disruptive", "inclusion", "equity"],
    "Key 4": ["global emissions", "mitigation pathways", "global temperature goal", "commitment"],
    "Key 5": ["domestic mitigation", "national targets", "NDC", "GHG emissions"],
    "Key 6": ["net zero", "phasing out fossil fuels", "supply-side measures", "demand-side measures"],
    "Key 7": ["just transition", "equitable", "participatory", "human rights", "youth"],
    "Key 8": ["economic diversification","different contexts", "green industrialization", "synergies"],
    "Key 9": ["adaptation", "reduce impacts", "threat of climate change"],
    "Key 10": ["adaptation plan", "adaptation action", "adaptation commitment"],
    "Key 11": ["local context", "local population", "transformational adaptation"],
    "Key 12": ["averting loss and damage", "addressing loss and damage", "limits to adaptation"],
    "Key 13": ["support for adaptation", "adaptation funding", "climate resilient development"],
    "Key 14": ["support for developing countries", "international public finance", "multilateral development banks"],
    "Key 15": ["financial flows", "investment", "public", "private", "international", "domestic"],
    "Key 16": ["cleaner technologies", "innovation", "development of new technologies", "technology transfer",],
    "Key 17": ["capacity-building", "country-led cooperation", "needs-based cooperation", "capacities are enhanced and retained"],
    }

# Language mappings for different tools.
TESSERACT2UN = {'eng': 'en', 'spa': 'es', 'fra': 'fr', 'rus': 'ru', 'ara': 'ar'}
NLTK2TESSERACT = {'en': 'eng', 'es': 'spa', 'fr': 'fra', 'ru': 'rus', 'ar': 'ara'}
