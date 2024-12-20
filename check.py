from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to generate predictions
def generate_response(input_text):
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate response
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
input_text = "I have fever and my knee hurts"
response = generate_response(input_text)
print(f"Input: {input_text}")
print(f"Response: {response}")