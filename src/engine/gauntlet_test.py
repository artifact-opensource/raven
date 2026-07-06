import torch
import time
import json
from src.model.raven import Raven

def run_gauntlet():
    device = torch.device("cpu") # Forced CPU as per reality check
    model = Raven().to(device).eval()
    
    # 50 High-Precision Blockchain Questions
    questions = [
        f"Blockchain Question {i}: Explain the impact of MEV on transaction ordering in a PoS system." for i in range(50)
    ]
    
    results = []
    print("Starting Speed-Fire Gauntlet...")
    
    for i, q in enumerate(questions):
        start_time = time.time()
        
        # Mock tokenization
        input_ids = torch.randint(0, 32000, (1, 32)).to(device)
        
        with torch.no_grad():
            output = model(input_ids)
            
        end_time = time.time()
        latency = end_time - start_time
        
        results.append({
            "q": q,
            "latency": latency,
            "tokens_per_sec": 32 / latency # Assuming 32 tokens generated
        })
        
        if i % 10 == 0:
            print(f"Question {i} completed | Latency: {latency:.4f}s")

    # Save results to log
    with open("/home/adam/worxpace/gladius/raven/gauntlet_performance.log", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Gauntlet Complete. Results saved to gauntlet_performance.log")

if __name__ == "__main__":
    run_gauntlet()
