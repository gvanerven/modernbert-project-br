import os
import torch
from datasets import load_dataset
from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from accelerate import Accelerator
from multiprocessing import cpu_count


def run_training():
    print('Starting')
    accelerator = Accelerator()

    BASE_MULTILANG_MODEL = "jhu-clsp/mmBERT-base" 
    TOKNZ_DIR_COMPL = f"{BASE_MULTILANG_MODEL}/256000/"
    

    WORK_DIR = os.getenv("WORK")
    #DATA_FOLDER = os.path.join(WORK_DIR, "data")
    CACHED_DATA_FOLDER = os.path.join(WORK_DIR, "cached_data")
    os.environ["HF_HOME"] = CACHED_DATA_FOLDER
    os.environ["TRITON_HIP_LLD_PATH"] = "/opt/rocm-7.1.0/lib/llvm/bin/ld.lld"
    os.chdir(WORK_DIR)

    accelerator.print(f"Working directory: {os.getcwd()}")

    tokenizer_name = os.path.join(WORK_DIR, "tokenizers", "MM", TOKNZ_DIR_COMPL)
    model_name = f"Modern/{BASE_MULTILANG_MODEL}-freeze-finetune-ptbr-exp"

    output_dir = f"training_test/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    #tokenized_datasets_name = os.path.join(
    #    DATA_FOLDER,
    #    f"unpadded-tokenized-for-training/custom/vocab_size:{vocabulary_size:_}/context_size:{context_size}",
    #)
    #tokenized_datasets = load_from_disk(tokenized_datasets_name)
    #training_dataset = tokenized_datasets['train']
    # eval_dataset = tokenized_datasets["test"]

    ds = load_dataset(
        "unb-labia/CCCPT-unpadded-tokenized-ModBertBR-vs32Kmxlen1K",
        split=["train","validation"],
        num_proc=max(1, cpu_count()-1),        
    )

    train_ds = ds[0]
    test_ds = ds[1]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=True, cache_dir=CACHED_DATA_FOLDER
    )

    config = ModernBertConfig.from_pretrained(
        BASE_MULTILANG_MODEL,
        attn_implementation="flash_attention_2",
    )

    model = ModernBertForMaskedLM(config=config)

    # Freezing weights
    for param in model.parameters():
        param.requires_grad = False

    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.3
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=4)

    random_eval_dataset = test_ds.shuffle(seed=42).select(range(1_500_000))

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        max_steps=500_000,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        dataloader_num_workers=32,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=10_000,
        save_strategy="steps",
        save_steps=10_000,
        save_total_limit=5,
        bf16=True,
        report_to=["tensorboard", "mlflow"],

        gradient_checkpointing=False,
        #torch_compile=True,
        
        per_device_eval_batch_size=64,
        eval_strategy="steps",
        eval_steps=10_000, 
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=random_eval_dataset,
        callbacks=[early_stopping],
        data_collator=data_collator,
    )

    accelerator.print("Starting training on all available GPUs...")
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    if last_checkpoint is not None:
        accelerator.print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        accelerator.print("No checkpoint found. Starting from scratch.")
        trainer.train()
    accelerator.print("Training complete!")

    trainer.save_model(f"saved_models/Modern/{BASE_MULTILANG_MODEL}-freeze-finetune-ptbr-expv1")

if __name__ == "__main__":
    run_training()
