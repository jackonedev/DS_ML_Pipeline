import os

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_PATH = os.path.join(PROJECT_ROOT, "datasets")
DATASET_NAME = "challenge_edMachina"

# Google Drive Download Variable Environment
SHARED_URL = os.getenv("SHARED_URL")
FILE_ID = SHARED_URL.split("/")[-2]
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
DOWNLOAD_PATH = os.path.join(DATASETS_PATH, DATASET_NAME + ".csv")

if not os.path.exists(os.path.join(PROJECT_ROOT, "datasets")):
    os.makedirs(os.path.join(PROJECT_ROOT, "datasets"))

if not os.path.exists(os.path.join(DATASETS_PATH, "parquet")):
    os.makedirs(os.path.join(DATASETS_PATH, "parquet"))

if not os.path.exists(os.path.join(DATASETS_PATH, "samples")):
    os.makedirs(os.path.join(DATASETS_PATH, "samples"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "reports")):
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "features")):
    os.makedirs(os.path.join(PROJECT_ROOT, "features"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "mlartifacts")):
    os.makedirs(os.path.join(PROJECT_ROOT, "mlartifacts"))


PARQUET_PATH = os.path.join(DATASETS_PATH, "parquet")
SAMPLES_PATH = os.path.join(DATASETS_PATH, "samples")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "features")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "mlartifacts")

# Makefile: new_attempt
if not os.path.exists(os.path.join(PROJECT_ROOT, "ds_versions")):
    os.makedirs(os.path.join(PROJECT_ROOT, "ds_versions"))
