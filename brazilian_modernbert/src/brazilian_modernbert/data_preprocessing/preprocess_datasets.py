import os
import logging
import nltk
from datasets import DatasetDict
from multiprocessing import cpu_count
import numpy as np
import math
from datasets import concatenate_datasets

from src.brazilian_modernbert.utils.text_helper import (
    get_document_metadata_paragraphs,
    get_document_metadata_entire_text,
)

nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("portuguese")

logger = logging.getLogger(__name__)


def clean_for_first_phase(dataset):
    cleaned_dataset = dataset.filter(
        lambda example: example["num_words"] >= 20
        and example["num_words"] <= 1024
        and example["stopwords"] >= 2
        and example["average"] >= 2
        and example["average"] <= 15
    )

    cleaned_dataset = cleaned_dataset.remove_columns(
        [col for col in cleaned_dataset.column_names if col != "text"]
    )  # only keep the 'text' column
    return cleaned_dataset


def clean_for_second_phase(dataset):
    # Data Engineering for Scaling Language Models to 128K Context https://arxiv.org/pdf/2402.10171
    # first phase has 44% of the initial dataset size
    # nr sentence aprox 8M to 5.5B tokens (estimate)
    cleaned_dataset_under_1024 = dataset["train"].filter(
        lambda example: example["num_words"] >= 20 # example["num_words"] >= 100 # mantendo dist inicial
        and example["num_words"] <= 1024
        and example["stopwords"] >= 2
        and example["average"] >= 2
        and example["average"] <= 15
    )

    len_dataset_u1024 = len(cleaned_dataset_under_1024)
    print(f"Size under 1024: {len_dataset_u1024}")
    s_size = math.ceil(len_dataset_u1024*0.09) # len_dataset_u1024*0.50*0.40 (almost 50B, changed to 0.07 and 0.16)
    print(f"Size sample under 1024: {s_size}")

    indices_under_1024 = np.random.choice(len(cleaned_dataset_under_1024), size=s_size, replace=False)

    cleaned_dataset_over_1024 = dataset["train"].filter(
        lambda example: example["num_words"] > 1024
    )

    len_dataset_o1024 = len(cleaned_dataset_over_1024)
    print(f"Size over 1024: {len_dataset_o1024}")
    # len_dataset_u1024*0.50*0.60
    indices_over_1024 = np.random.choice(len_dataset_o1024, size=math.ceil(len_dataset_u1024*0.13), replace=True)

    print(f"Size sample over 1024: {len(indices_over_1024)}")
    cleaned_dataset = concatenate_datasets([cleaned_dataset_under_1024.select(indices_under_1024), cleaned_dataset_over_1024.select(indices_over_1024)])

    cleaned_dataset = cleaned_dataset.remove_columns(
        [col for col in cleaned_dataset.column_names if col != "text"]
    )  # only keep the 'text' column
    return cleaned_dataset


def preprocess_concatenated_dataset(data_path, dataset):
    logger.info("Preprocessing concatenated dataset")

    #preprocessed_dataset = dataset['train'].map(
    #    get_document_metadata_paragraphs,
    #    batched=True,
    #    remove_columns=["text"],
    #    num_proc=max(1, cpu_count()-1),
    #)
    preprocessed_dataset = dataset.map(
        get_document_metadata_entire_text,
        batched=True,
        #remove_columns=["text"],
        num_proc=max(1, cpu_count()-1),
    )

    #preprocessed_dataset = preprocessed_dataset.rename_column(
    #    "paragraphs", "text"
    #)

    #logger.info("Cleaning for first training phase")
    logger.info("Cleaning for second training phase")
    # cleaned_for_fist_phase = clean_for_first_phase(preprocessed_dataset)
    cleaned_for_second_phase = clean_for_second_phase(preprocessed_dataset)

    #logger.info("Splitting dataset")
    #split_dataset = cleaned_for_fist_phase.train_test_split(
    #    test_size=0.1, shuffle=True, seed=42
    #)
    # split_dataset = cleaned_for_second_phase.train_test_split(
    #     test_size=0.1, shuffle=True, seed=42
    # )

    split_dataset = DatasetDict({
        'train': cleaned_for_second_phase,
        'test': dataset['test'],
        'validation': dataset['validation'] # The 'train' split of the second split is the validation set
    })

    train_count = len(split_dataset["train"])
    logger.info(f"Total Training Samples available: {train_count:_}")

    split_save_path = os.path.join(data_path, "split_datasets")
    split_dataset.save_to_disk(split_save_path)
    logger.info("Splitted dataset saved on %s", split_save_path)

    return split_dataset
