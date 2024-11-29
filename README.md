# Lunar-v2-lite

A lightweight language model with only 70M parameters, designed for efficient natural language processing while maintaining good performance.

## Model Architecture
- 6 transformer layers
- 256 hidden dimensions
- 8 attention heads
- 1024 intermediate size
- Approximately 70M parameters

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```
This will train the model on the WikiText-2 dataset and save checkpoints in the `checkpoints` directory.

3. Chat with the model:
```bash
python chat.py
```
This will load the latest checkpoint and start an interactive chat session.

## Features
- Efficient transformer architecture
- Top-k sampling for text generation
- Interactive chat interface
- Checkpoint saving and loading
- Training progress tracking

## Training Data
The model is trained on the WikiText-2 dataset, which contains high-quality Wikipedia articles.

## Note
This is a lightweight model designed for educational purposes and basic text generation tasks. While it has fewer parameters than larger models, it can still engage in meaningful conversations and generate coherent text.
