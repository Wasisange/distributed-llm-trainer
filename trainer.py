import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class LLMTrainer:
    def __init__(self, model, optimizer, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.rank = 0
        self.world_size = 1

        if torch.cuda.is_available():
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.model = DDP(self.model.to(self.rank), device_ids=[self.rank])

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                if torch.cuda.is_available():
                    data, target = data.to(self.rank), target.to(self.rank)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()
                self.optimizer.step()

                if self.rank == 0 and batch_idx % 10 == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

    def save_checkpoint(self, path):
        if self.rank == 0:
            torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
