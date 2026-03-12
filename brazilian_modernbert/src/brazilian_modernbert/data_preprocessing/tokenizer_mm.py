from datasets import load_dataset
from multiprocessing import cpu_count
from transformers import AutoTokenizer
import os
import logging

from src.brazilian_modernbert.configs import (
    WORK_DIR,
    BASE_MULTILANG_MODEL,
)

from src.brazilian_modernbert.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def make_tokenizer():
    logger.info(f"Creating Tokenizer from {BASE_MULTILANG_MODEL}")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MULTILANG_MODEL)

    ds = load_dataset(
        "unb-labia/CCCPT-splited_preprocessed_max1024sz_sentences",
        split="train",
        num_proc=max(1, cpu_count()-1),        
    )

    ds = ds.shuffle(seed=42).select(range(5_000_000))
    logger.info(f"Vocab Size {base_tokenizer.vocab_size}")
    tokenizer = base_tokenizer.train_new_from_iterator(ds['text'], base_tokenizer.vocab_size)
    tokenizer.save_pretrained(os.path.join(WORK_DIR, "tokenizers", "MM", BASE_MULTILANG_MODEL, str(base_tokenizer.vocab_size)))
    logger.info(f"Tokenizer created. Len: {len(base_tokenizer)}")

if __name__ == "__main__":
    make_tokenizer()