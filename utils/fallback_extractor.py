import os
from random import shuffle
from stocktake.text_extractor import extraction_data_from_image

from config import DATA_FOLDER


if __name__ == '__main__':
    pdf_folder = os.path.join(DATA_FOLDER, 'pdfs')
    pdf_files = os.listdir(pdf_folder)
    md_sums = [file.split('.')[0] for file in pdf_files]
    # Shuffle and only extract from at most 32 submissions.
    shuffle(md_sums)
    md_sums = md_sums[:32]
    # See how to deal with it.
    for md_sum in md_sums:
        try:
            extraction_data_from_image(md_sum)
        except:
            print(f'Failure at {md_sum}.')

    print('YOU ARE TERMINATED!')