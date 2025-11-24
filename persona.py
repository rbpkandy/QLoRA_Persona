import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

#Setup and Constants
BASE_MODEL = config["model"]["base_model"]
NEW_MODEL = config["model"]["new_model"]
SYSTEM_PROMPT = config["training"]["system_prompt"]
DATASET_FILE = config["data"]["dataset_file"]
NUM_EPOCHS = config["training"]["num_epochs"]
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
WEIGHT_DECAY = config["training"]["weight_decay"]
LOGGING_STEPS = config["training"]["logging_steps"]
MAX_GRAD_NORM = config["training"]["max_grad_norm"]
WARMUP_RATIO = config["training"]["warmup_ratio"]
GROUP_BY_LENGTH = config["training"]["group_by_length"]
BF16_ENABLED = config["training"]["bf16_enabled"]

LORA_R = config["lora"]["r"]
LORA_ALPHA = config["lora"]["alpha"]
LORA_DROPOUT = config["lora"]["dropout"]

def format_instruction_data(examples):
    formatted_texts = []
    
    for instruction, response in zip(examples["instruction"], examples["response"]):
        formatted_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        
        formatted_texts.append(formatted_text)
    
    return {"text": formatted_texts}

raw_data = {
    "instruction": [
        "What is the status of my refund for order #12345?",
        "My package is delayed by three days. I am very frustrated.",
        "Can I change my subscription plan to the premium tier?",
    ],
    "response": [
        "I'd be happy to check that for you! I see order #12345 was marked for refund on Tuesday. You should see the credit reflected in your account within 3-5 business days. Is there anything else I can assist you with today?",
        "I sincerely apologize for the shipping delay you're experiencing. I understand this is frustrating. I've personally escalated your tracking to our shipping partners. Thank you for your patience; please let me know if you need anything else!",
        "Absolutely! Changing your subscription to the premium tier is easy. I've initiated the upgrade, which will take effect immediately. Do you have any other questions about your new plan features?",
    ],
}

dataset = load_dataset("csv", data_files={"train": DATASET_FILE})

#Model and Tokenizer Setup

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = dataset.map(lambda x: {"text": [tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": u},
    {"role": "assistant", "content": a},
], tokenize=False) for u, a in zip(x["instruction"], x["response"])]}, batched=True, remove_columns=["instruction", "response"])


#LoRA and Training Configuration

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=False,
    bf16=BF16_ENABLED,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=GROUP_BY_LENGTH,
    lr_scheduler_type="constant",
)

#Trainer and Execution

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
)

print("Starting fine tuning")
trainer.train()
print("Fine-tuning complete")

trainer.model.save_pretrained(NEW_MODEL)
tokenizer.save_pretrained(NEW_MODEL)
