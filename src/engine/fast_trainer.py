import torch
from ..model.raven import Raven

class RavenTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # Will upgrade to MuonClip
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_step(self, batch):
        self.optimizer.zero_grad()
        logits = self.model(batch['input_ids'])
        loss = self.criterion(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == "__main__":
    print("Fast Trainer: Ready for continuous synthetic data ingestion.")
