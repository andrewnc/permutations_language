from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, RobertaConfig, RobertaForMaskedLM
import torch
from fast_soft_trainer import FastSoftTrainer

config = RobertaConfig(
    vocab_size=30_522,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
GLUE_TASKS = ['cola', 'mnli', 'mnli-mm', 'mrcp', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

task = GLUE_TASKS[0]
model_checkpoint = 'distilbert-base-uncased'
batch_size = 16

dataset = load_dataset("glue", task)
metric = load_metric("glue", task)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

num_labels = 3 if task.startswith("mnli") else 1 if task =="stsb" else 2
model = RobertaForMaskedLM(config)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# TODO(andrew) finish this fast soft trainer class by building the proper loss function
trainer = FastSoftTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)