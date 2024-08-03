import os

import gdown

from utils.config import DOWNLOAD_PATH, DOWNLOAD_URL


def download_dataset() -> None:
    if not os.path.exists(DOWNLOAD_PATH):
        gdown.download(DOWNLOAD_URL, DOWNLOAD_PATH, quiet=False)
        print(f"File downloaded correctly and \nsaved on: {DOWNLOAD_PATH}")
        print(f"Archivo descargado correctamente y \nguardado en: {DOWNLOAD_PATH}")
    else:
        print(f"{os.path.basename(DOWNLOAD_PATH)} exists, moving on.")
        print(f"{os.path.basename(DOWNLOAD_PATH)} existe, continuamos.")
