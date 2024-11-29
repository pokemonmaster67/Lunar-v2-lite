import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from lunar_model import LunarConfig, LunarV2Lite
from tqdm import tqdm
import os

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def train():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = dataset['text']
    texts = [text for text in texts if len(text.strip()) > 0]  # Remove empty lines

    # Create dataset
    train_dataset = TextDataset(texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize model
    config = LunarConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    
    model = LunarV2Lite(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0)

            optimizer.zero_grad()
            outputs = model(input_ids, position_ids=position_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, config.vocab_size), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (len(progress_bar) + 1)})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1} average loss: {avg_loss:.4f}')

        # Save checkpoint
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'{checkpoint_dir}/lunar_v2_lite_epoch_{epoch + 1}.pt')

if __name__ == "__main__":
    train()
