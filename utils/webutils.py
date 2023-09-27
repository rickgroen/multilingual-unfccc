import os
import requests

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


def download(url: str, dest_folder: str, filename: str, second_attempt: bool = False):
    """ The web portal for downloading submissions can be fickle. We attempt to download at least twice.
    """
    file_path = os.path.join(dest_folder, filename)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
    except:
        print(f'Failed to retrieve {url} within the timeout.')
        if not second_attempt:
            download(url, dest_folder, filename, second_attempt=True)
        return False
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            f.write(r.content)
        return True
    else:
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))
        return False
