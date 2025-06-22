import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 모델
model_name = "meta-llama/Llama-2-7b-hf"  # RunPod에서는 계정 토큰 필요함
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 모델 로드 + LoRA 적용
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 데이터셋 로드
dataset = load_dataset("json", data_files="../data/train.jsonl")["train"]

def format_instruction(example):
    return f"{example['instruction']}\n{example['input']}\n"

def preprocess(example):
    full_prompt = format_instruction(example)
    input_ids = tokenizer(full_prompt + example["output"], truncation=True, padding="max_length", max_length=512)
    input_ids["labels"] = input_ids["input_ids"].copy()
    return input_ids

tokenized_dataset = dataset.map(preprocess)

# 학습 설정
training_args = TrainingArguments(
    output_dir="../models",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=1,
    report_to="none"
)


trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 학습 시작
trainer.train()

# LoRA adapter 저장
model.save_pretrained("../models/adapter")