# Distributed LLM Trainer

A high-performance framework for training large language models across multiple GPUs using PyTorch and DeepSpeed.

## Features
- Distributed training with DeepSpeed
- Mixed-precision training
- Gradient accumulation
- Checkpointing and resuming training

## Installation

```bash
git clone https://github.com/Wasisange/distributed-llm-trainer.git
cd distributed-llm-trainer
pip install -r requirements.txt
```

## Usage

```python
import torch
from trainer import LLMTrainer

# Initialize model, optimizer, and data loader
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters())
dataloader = [(torch.randn(10), torch.randn(10)) for _ in range(10)]

trainer = LLMTrainer(model, optimizer, dataloader)
trainer.train(num_epochs=5)
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
