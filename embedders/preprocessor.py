import copy
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re

from config import SENTENCE_MIN_WORDS, SENTENCE_MAX_WORDS, SENTENCE_MIN_CHARACTERS, \
    MIN_SENTENCES_PER_PARAGRAPH, SPLIT_PARAGRAPH_WORDS


class DocumentPreprocessor:
    language_map = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'ru': 'russian', 'ar': 'arabic'}
    stop_words = set(stopwords.words('english'))

    @staticmethod
    def _sentence_tokenize(messy_text, language):
        # NLTK does not offer arabic sentence tokenization, and online implementations do so on word basis.
        if not language == 'arabic':
            sentences = sent_tokenize(messy_text, language=language)
        else:
            tokenized_sentences = sent_tokenize(messy_text)
            sentences = []
            for initial_sentence in tokenized_sentences:
                sentences.extend(initial_sentence.split('.'))
            sentences = [sentence for sentence in sentences if len(sentence) > 0]
        # Tokenize deals poorly with e.g. or i.e.
        for idx in range(len(sentences) - 2, -1, -1):
            if sentences[idx][-4:] == 'e.g.' or sentences[idx][-4:] == 'i.e.':
                sentences[idx] = f'{sentences[idx]} {sentences[idx + 1]}'
                sentences.pop(idx + 1)
        return sentences


class TesseractDocumentPreprocessor(DocumentPreprocessor):

    def __init__(self, page_dict):
        self.language = self.language_map[page_dict['language']]
        del page_dict['language']
        self.pages = [page_dict[key] for key in sorted(page_dict.keys())]
        self.relevant_paragraphs = []

    @staticmethod
    def _is_contents_page(paragraph):
        low_paragraph = copy.deepcopy(paragraph).lower()
        english_check = (low_paragraph == 'table of contents' or low_paragraph == 'contents')
        spanish_check = (low_paragraph == 'tabla de contenido' or low_paragraph == 'contenidos')
        french_check = (low_paragraph == 'table des matières' or low_paragraph == 'contenu')
        russian_check = low_paragraph == 'оглавление'
        return english_check or spanish_check or french_check or russian_check

    @staticmethod
    def _is_references_page(paragraph):
        low_paragraph = copy.deepcopy(paragraph).lower()
        return low_paragraph == 'references' or low_paragraph == 'referencias' or  low_paragraph == 'références' or  low_paragraph == 'Рекомендации'

    def reduce_contents_references(self):
        table_of_content_page, references_page = 0, -1
        for key, page in enumerate(self.pages):
            if any([self._is_contents_page(paragraph) for paragraph in page]):
                table_of_content_page = key
            if any([self._is_references_page(paragraph) for paragraph in page]):
                references_page = key
        if references_page > 0:
            self.pages = self.pages[table_of_content_page:references_page]
        else:
            self.pages = self.pages[table_of_content_page:]

    def process(self):
        # Do not include pages before table of contents and after references.
        self.reduce_contents_references()
        # Now loop over the pages and remove those candidates that do not fit our definition of a paragraph.
        for page in self.pages:
            for candidate in page:
                self.process_candidate(candidate)
        return self.relevant_paragraphs

    def process_candidate(self, candidate):
        # Deal with the new line character and the hyphens.
        candidate = candidate.replace('- ', '-')
        # Remove urls, e-mail.
        candidate = re.sub(r'http\S+', '', candidate)
        candidate = re.sub(r'\S+@\S+(?:\.\S+)+', '', candidate)
        # Remove certain enumeration artefacts before the paragraph.
        candidate = self._remove_enumeration_sign(candidate)
        # If the start of the sentence is Table, or annex or figure, drop this candidate.
        if candidate[:5].lower() == 'table' or candidate[:5].lower() == 'annex' or candidate[:6].lower() == 'figure':
            return None
        # Tokenize based on language.
        tokenized_sentences = self._sentence_tokenize(candidate, self.language)
        if len(tokenized_sentences) == 1:
            return None
        # Remove sentences if more than 10% of the sentence is numeric or 20% is capped.
        tokenized_sentences = [sentence for sentence in tokenized_sentences
                               if (len(''.join(re.findall(r'\d+', sentence))) / len(sentence)) < 0.1]
        tokenized_sentences = [sentence for sentence in tokenized_sentences
                               if (len(''.join(re.findall(r'[A-Z]', sentence))) / len(sentence)) < 0.2]
        # Remove short and long sentences.
        tokenized_sentences = [sentence for sentence in tokenized_sentences
                               if (len(sentence) >= SENTENCE_MIN_CHARACTERS and len(sentence.split()) >= SENTENCE_MIN_WORDS)]
        tokenized_sentences = [sentence for sentence in tokenized_sentences
                               if len(sentence.split()) < SENTENCE_MAX_WORDS]
        # If there are less than 2 sentences.
        if len(tokenized_sentences) < MIN_SENTENCES_PER_PARAGRAPH:
            return None
        paragraph = ' '.join(tokenized_sentences)
        # To use distiluse, we have to have max 128 tokens (or the remainder will be cut).
        if len(paragraph.split()) > 100:
            return self._split_paragraphs_to_fit_distiluse(paragraph)
        self.relevant_paragraphs.append(paragraph)

    def _split_paragraphs_to_fit_distiluse(self, candidate):
        retokenized_sentences = self._sentence_tokenize(candidate, self.language)
        # Drop the final sentence if it is not a full one.
        if retokenized_sentences[-1][-1] not in ['.', '?', '!']:
            retokenized_sentences.pop(-1)
            if sum([len(p.split()) for p in retokenized_sentences]) <= SPLIT_PARAGRAPH_WORDS:
                self.relevant_paragraphs.append(' '.join(retokenized_sentences))
                return
        # Then start splitting the paragraph in smaller paragraphs.
        smaller_paragraph = ''
        for sentence in retokenized_sentences:
            if len(smaller_paragraph.split()) + len(sentence.split()) >= SPLIT_PARAGRAPH_WORDS:
                self.relevant_paragraphs.append(smaller_paragraph)
                smaller_paragraph = ''
            smaller_paragraph += f' {sentence}' if len(smaller_paragraph) > 0 else sentence
        self.relevant_paragraphs.append(smaller_paragraph)
        return

    @staticmethod
    def _remove_enumeration_sign(sentence):
        if sentence[:2] in ['* ', '© ', '@ ', '+ ', '» ', '= ', '"*']:
            return sentence[2:]
        return sentence
