"""
OPTIONAL — Phase 2 fine-tune (run on Kaggle's free GPU).

You do NOT need this to ship. The product works on the pretrained model + RAG.
This is the paid upgrade that (a) makes answers sound like Hubmicroo's own brand
voice and (b) lets you honestly tell the client: "I fine-tuned the model on your
store's data."

It does QLoRA fine-tuning of a small open model on a JSONL of Hubmicroo
question/answer pairs. QLoRA = train tiny adapter weights in 4-bit, so it fits in
Kaggle's free 16GB T4.

HOW TO RUN ON KAGGLE
  1. New Notebook -> Settings -> Accelerator: GPU T4 x2 (or P100).
  2. Upload your data as `hubmicroo_qa.jsonl` (format below) to the notebook.
  3. pip install -q transformers peft trl bitsandbytes accelerate datasets
  4. Run this file. The adapter is saved to ./hubmicroo-lora.
  5. Download the adapter, merge it, and serve with Ollama (see README, Phase 2).

DATA FORMAT (hubmicroo_qa.jsonl) — one JSON object per line:
  {"instruction": "Do you deliver to Lahore?", "response": "Yes! Orders in Lahore arrive in 2-3 working days, free above 5000 PKR."}
  {"instruction": "کیا بلوٹوتھ ہیڈ فون دستیاب ہے؟", "response": "جی ہاں، ساؤنڈ میکس وائرلیس ہیڈ فون 4500 روپے میں دستیاب ہے۔"}
Generate 300-800 of these from the product catalogue + real customer chats.
"""
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"   # matches the served model
DATA_FILE = "hubmicroo_qa.jsonl"
OUT_DIR = "hubmicroo-lora"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto"
)

ds = load_dataset("json", data_files=DATA_FILE, split="train")


def to_chat(row):
    messages = [
        {"role": "system", "content": "You are Hubmicroo's helpful shopping assistant."},
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


ds = ds.map(to_chat, remove_columns=ds.column_names)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        max_seq_length=1024,
    ),
)

trainer.train()
trainer.save_model(OUT_DIR)
print(f"Done. LoRA adapter saved to ./{OUT_DIR} — download and merge it (see README).")
