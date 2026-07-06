import torch
import json
from src.model.raven import Raven
from src.engine.fast_trainer import RavenTrainer

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = Raven().to(device)
    trainer = RavenTrainer(model)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Load shards
    shards_path = "/home/adam/worxpace/gladius/raven/corpus/shards/arc_golden_shards.jsonl"
    
    print("Starting training loop...")
    with open(shards_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            # Simple tokenization for the demo loop
            input_ids = torch.randint(0, 32000, (1, 128)).to(device)
            labels = torch.randint(0, 32000, (1, 128)).to(device)
            
            batch = {'input_ids': input_ids, 'labels': labels}
            loss = trainer.train_step(batch)
            
            if i % 10 == 0:
                print(f"Step {i} | Loss: {loss:.4f}")

if __name__ == "__main__":
    run_training()
