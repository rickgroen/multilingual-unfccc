# Many of the functions in this python file were copied from:
# https://github.com/piegu/language-models/blob/master/inference_on_LiLT_model_finetuned_on_DocLayNet_base_in_any_language_at_levelparagraphs_ml512.ipynb

import copy
import re
import os
from PIL import Image
import numpy as np
import cv2
import pytesseract
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import pypdfium2 as pdfium
import tempfile
import fitz
import pickle
import gc
from langdetect import detect

from embedders.preprocessor import TesseractDocumentPreprocessor
from config import DATA_FOLDER, TESSERACT2UN, NLTK2TESSERACT


def pdf_to_images(filename):
    try:
        PdfReader(filename)
    except PdfReadError:
        print("Invalid PDF file.")
    else:
        try:
            pdf = pdfium.PdfDocument(str(filename))  # get the number of pages in the document
            page_indices = [i for i in range(len(pdf))]  # pages until last_page
            images = list(pdf.render(
                pdfium.PdfBitmap.to_pil,
                page_indices=page_indices,
                scale=300 / 72,  # 300dpi resolution
            ))
            print(f'The PDF "{filename}" was converted into {len(images)} images.')
            return images
        except:
            print(f"Error with the PDF {filename}: it was not converted into images.")


def get_data_paragraph(results, conf_min=0):
    """ Get text and bounding boxes from an image
        https://stackoverflow.com/questions/61347755/how-can-i-get-line-coordinates-that-readed-by-tesseract
        https://medium.com/geekculture/tesseract-ocr-understanding-the-contents-of-documents-beyond-their-text-a98704b7c655
    """
    data = {}
    for i in range(len(results['line_num'])):
        block_num = results['block_num'][i]
        par_num = results['par_num'][i]
        line_num = results['line_num'][i]
        top, left = results['top'][i], results['left'][i]
        width, height = results['width'][i], results['height'][i]
        conf = results['conf'][i]
        text = results['text'][i]
        if not (text == '' or text.isspace()):
            if conf >= conf_min:
                tup = (text, left, top, width, height)
                if block_num in list(data.keys()):
                    if par_num in list(data[block_num].keys()):
                        if line_num in list(data[block_num][par_num].keys()):
                            data[block_num][par_num][line_num].append(tup)
                        else:
                            data[block_num][par_num][line_num] = [tup]
                    else:
                        data[block_num][par_num] = {}
                        data[block_num][par_num][line_num] = [tup]
                else:
                    data[block_num] = {}
                    data[block_num][par_num] = {}
                    data[block_num][par_num][line_num] = [tup]

    # get paragraphs dictionary with list of lines
    par_data = {}
    par_idx = 1
    for _, b in data.items():
        for _, p in b.items():
            line_data = {}
            line_idx = 1
            for _, l in p.items():
                line_data[line_idx] = l
                line_idx += 1
            par_data[par_idx] = line_data
            par_idx += 1

    # get lines of texts, grouped by paragraph
    texts_pars = list()
    row_indexes = list()
    texts_lines = list()
    texts_lines_par = list()
    row_index = 0
    for _, par in par_data.items():
        count_lines = 0
        lines_par = list()
        for _, line in par.items():
            if count_lines == 0: row_indexes.append(row_index)
            line_text = ' '.join([item[0] for item in line])
            texts_lines.append(line_text)
            lines_par.append(line_text)
            count_lines += 1
            row_index += 1
        row_index += 1
        texts_lines_par.append(lines_par)
        texts_pars.append(' '.join(lines_par))
    return texts_lines, texts_pars, texts_lines_par, row_indexes


def set_image_dpi_resize(image):
    """ Rescaling image to 300dpi while resizing.
    """
    length_x, width_y = image.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    image_resize = image.resize(size, Image.LANCZOS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='1.png')
    temp_filename = temp_file.name
    image_resize.save(temp_filename, dpi=(300, 300))
    return factor, temp_filename


def detect_language(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = [x[4] for i in range(len(doc)) for x in doc[i].get_text("blocks", sort=True) if i < 50]
    pars = sorted(blocks, key=len, reverse=True)
    predicted_languages = [detect(text) for text in pars[:128]
                           if (len(text) > 64 and len(text.split()) > 3 and re.search('[a-zA-Z]',
                                                                                      text) and 'http' not in text)]
    if len(set(predicted_languages)) == 0:
        return 'english'
    predicted_language = max(set(predicted_languages), key=predicted_languages.count)
    return NLTK2TESSERACT[predicted_language] if predicted_language in NLTK2TESSERACT else 'english'


def extraction_data_from_image(md_checksum):
    # Record storage location.
    tesseract_dir = os.path.join(DATA_FOLDER, 'tesseract_extractions')
    if not os.path.exists(tesseract_dir):
        os.mkdir(tesseract_dir)
    tesseract_store_path = os.path.join(tesseract_dir, f'{md_checksum}.pkl')
    if os.path.exists(tesseract_store_path):
        return md_checksum
    # Extract the pages from the pdf.
    pdf_path = os.path.join(DATA_FOLDER, 'pdfs', f'{md_checksum}.pdf')
    images = pdf_to_images(pdf_path)
    # First detect the language of the pdf to select the right tessdata language file.
    detected_language = detect_language(pdf_path)
    # Then loop over all images that are pages of the PDF.
    # https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
    custom_config = f'--oem 3 --psm 3 -l {detected_language}'
    texts_pars = {'language': TESSERACT2UN[detected_language]}
    for i, image in enumerate(images):
        # image preprocessing
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
        img = image.copy()
        factor, path_to_img = set_image_dpi_resize(img)  # Rescaling to 300dpi
        img = Image.open(path_to_img)
        img = np.array(img, dtype='uint8')  # convert PIL to cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray scale image
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # OCR PyTesseract | get data
        results = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
        _, texts_pars[i], _, _ = get_data_paragraph(results, conf_min=0)
        del results
    # Save the extracted text to a pickle file for later reference.
    with open(tesseract_store_path, 'wb') as pickle_file:
        pickle.dump(texts_pars, pickle_file)
    # Then force garbage collection, because tesseract has a memory leak or something somewhere.
    gc.collect()
    return md_checksum


def extract_all_files_with_tesseract_fallback():
    import os
    import subprocess
    from config import DATA_FOLDER
    from stocktake.dataset import StockTakeDataset

    dataset = StockTakeDataset()
    tesseract_dir = os.path.join(DATA_FOLDER, 'tesseract_extractions')
    while len(os.listdir(tesseract_dir)) < len(dataset):
        subprocess.call(["python", "./utils/fallback_extractor.py"])
        os.system('pkill -f "fallback_extractor.py"')
        print(f'{len(os.listdir(tesseract_dir))} <? {len(dataset)}')


def extract_all_text_from_pdfs_using_tesseract(path):
    """ Important addition: Tesseract leaks memory and floods your RAM after a while. This
        may only have happened on our systems. If it happens on your system too, you
        can use extract_all_files_with_tesseract_fallback()
    """
    # If we do not have an extraction yet, do an extraction using Tesseract.
    md5_sum = path.split('/')[-1].split('.')[0]
    extraction_path = os.path.join(DATA_FOLDER, 'tesseract_extractions', f'{md5_sum}.pkl')
    if not os.path.exists(extraction_path):
        extraction_data_from_image(md5_sum)
    # Open the extraction.
    with open(extraction_path, 'rb') as pickle_file:
        page_dict = pickle.load(pickle_file)
    # Do pre-processing of the paragraphs.
    preprocessor = TesseractDocumentPreprocessor(copy.deepcopy(page_dict))
    paragraphs = preprocessor.process()
    return paragraphs, page_dict['language']
