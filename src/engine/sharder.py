import json
import os
import glob

def create_shards(corpus_dir, output_dir):
    files = glob.glob(f"{corpus_dir}/*.md")
    dataset = []
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Basic structural sharding: Split by headers or paragraphs
            # For high-fidelity, we wrap the content in an instruction/response pair
            dataset.append({
                "instruction": f"Analyze the following ARC Protocol specification from {os.path.basename(file)}",
                "response": content
            })
    
    # Save as JSONL shards
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/arc_golden_shards.jsonl", 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Successfully sharded {len(files)} files into arc_golden_shards.jsonl")

if __name__ == "__main__":
    create_shards("/home/adam/worxpace/gladius/raven/corpus", "/home/adam/worxpace/gladius/raven/corpus/shards")
