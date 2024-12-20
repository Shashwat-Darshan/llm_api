from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_response(input_text, checkpoint_path):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Format input with prompt
    prompt = f"User: {input_text}\nAssistant: Based on your symptoms, you may have "
    
    # Tokenize with attention mask
    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100
    )
    
    # Move to device
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Generate response with fixed parameters
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,  # Enable sampling
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from response
    if "Assistant: Based on your symptoms, you may have" in response:
        response = response.split("Assistant: Based on your symptoms, you may have")[-1].strip()
    
    return response

# Test with example input
input_text = "I am having problem with skin rash and joint pain"
checkpoints = ["./fine_tuned_model/checkpoint-1600", "./fine_tuned_model/checkpoint-1800", "./fine_tuned_model/checkpoint-1845"]

for checkpoint in checkpoints:
    response = generate_response(input_text, checkpoint)
    print(f"Checkpoint: {checkpoint}")
    print(f"Input: {input_text}")
    print(f"Response: Possible condition: {response}\n")