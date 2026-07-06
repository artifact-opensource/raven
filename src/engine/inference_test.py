import torch
from src.model.raven import Raven

def test_raven():
    model = Raven()
    model.eval()
    
    # Mock input IDs for the prompt
    input_ids = torch.randint(0, 32000, (1, 32)) 
    
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"Inference successful. Output tensor shape: {output.shape}")
    print("Raven is responding. Analyzing logic...")

if __name__ == "__main__":
    test_raven()
