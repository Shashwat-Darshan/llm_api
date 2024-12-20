import json
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Step 1: Load dataset
with open("symptoms_dataset.json", "r") as file:
    data = json.load(file)

# Step 2: Prepare dataset with prompt engineering
def prepare_data(data):
    formatted_data = []
    for item in data:
        # Format input and output as conversation
        input_text = f"User: {item['input']}\nAssistant: Based on your symptoms, you may have "
        output_text = f"{item['output']}"
        formatted_data.append({
            "input": input_text,
            "output": output_text
        })
    return formatted_data

formatted_data = prepare_data(data)
dataset = Dataset.from_dict({
    "input": [item["input"] for item in formatted_data],
    "output": [item["output"] for item in formatted_data]
})

# Step 3: Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Step 4: Preprocess function
def preprocess_function(examples):
    # Combine input and output
    texts = [input_text + output_text for input_text, output_text in zip(examples["input"], examples["output"])]
    
    # Tokenize with padding and truncation
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings

# Process dataset
train_dataset = dataset.map(preprocess_function, batched=True)

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    save_steps=200,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=100,
    fp16=True,
    save_total_limit=3,
)


# Step 6: Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Step 7: Train with checkpoint resume
checkpoint_path = "./fine_tuned_model/checkpoint-1400"
trainer.train(resume_from_checkpoint=checkpoint_path)

# Save the model
trainer.save_model("./fine_tuned_model")