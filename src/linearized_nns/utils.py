import os
import shutil
import zipfile
import urllib.request


def download_repo(url, save_to):
    zip_filename = save_to + '.zip'
    urllib.request.urlretrieve(url, zip_filename)

    if os.path.exists(save_to):
        shutil.rmtree(save_to)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    del zip_ref
    assert os.path.exists(save_to)
