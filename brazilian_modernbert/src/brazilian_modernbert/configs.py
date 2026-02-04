import os
import logging

logger = logging.getLogger(__name__)

LOAD_AND_PREPROCESS_DATASET = True
TRAIN_TOKENIZER = False

WORK_DIR = '/home/gvanerven/code' #os.getenv("WORK")
DATA_FOLDER = os.path.join(WORK_DIR, "data")
CACHED_DATA_FOLDER = os.path.join(WORK_DIR, "data", "tmpdata")
USE_ALTERNATIVE_TOKENIZER = False
ALTERNATIVE_TOKENIZER = "neuralmind/bert-large-portuguese-cased"


VOCABULARY_SIZE = 32_768
CONTEXT_SIZE = 1024

# Salvamos o path do Cache par ao HuggingFace
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# os.environ["HF_HOME"] = CACHED_DATA_FOLDER
# os.environ["HF_DATASETS_CACHE"] = CACHED_DATA_FOLDER
# os.environ["TRANSFORMERS_CACHE"] = CACHED_DATA_FOLDER

os.chdir(WORK_DIR)

logger.info("Working directory set: %s", os.getcwd())
