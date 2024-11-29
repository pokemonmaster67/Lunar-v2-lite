import torch
from transformers import GPT2Tokenizer
from lunar_model import LunarConfig, LunarV2Lite
import os

def load_model(checkpoint_path):
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model with same config as training
    config = LunarConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    
    model = LunarV2Lite(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer

def chat():
    # Find latest checkpoint
    checkpoint_dir = "checkpoints"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoints found. Please train the model first.")
        return
    
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print("\nLunar-v2-lite Chat Interface")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Tokenize input
        input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)
        
        # Generate response
        with torch.no_grad():
            output_sequence = model.generate(
                input_ids=input_ids,
                max_length=50,  # Adjust as needed
                temperature=0.7,
                top_k=50
            )
        
        # Decode and print response
        response = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
        print(f"Lunar: {response}")

if __name__ == "__main__":
    chat()
