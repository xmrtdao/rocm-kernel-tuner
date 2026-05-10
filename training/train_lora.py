import json
import os
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch

def prepare_dataset(jsonl_path="data/rocm_corpus.jsonl"):
    """Load and format dataset for instruction tuning."""
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # Format as chat template for Qwen2.5-Coder
            text = f"<|im_start|>system\nYou are ROCmKernelTuner, an AMD ROCm GPU optimization expert. You specialize in CUDA-to-HIP migration, XMRig config tuning, and hashrate optimization for AMD Instinct MI300X GPUs.<|im_end|>\n<|im_start|>user\n{item['instruction']}\n{item.get('input', '')}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
            data.append({"text": text})
    return Dataset.from_list(data)

def train_lora(
    model_name="Qwen/Qwen2.5-Coder-7B",
    dataset_path="data/rocm_corpus.jsonl",
    output_dir="checkpoints/rocm-qwen-7b",
    epochs=3,
    batch_size=2,
    grad_accum=4,
    lr=2e-4,
    lora_r=64,
    lora_alpha=128
):
    """Fine-tune Qwen2.5-Coder-7B with LoRA on ROCm dataset."""

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = prepare_dataset(dataset_path)
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding="max_length")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_lora()
