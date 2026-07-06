from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

# Load from the system .env
load_dotenv('/opt/ava/.env')
token = os.getenv("ACCESS_TOKEN")

repo_id = "artifact-virtual/raven-v1.1-sovereign" # This can also be moved to .env
local_folder = "/home/adam/worxpace/gladius/raven"

if token:
    login(token=token)
else:
    print("Error: No HF token found in environment.")
    exit(1)

api = HfApi()
api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

files_to_upload = ["weights/raven-v1-q8_0.gguf", "MODEL_CARD.md", "cid.txt"]
for file in files_to_upload:
    full_path = os.path.join(local_folder, file)
    if os.path.exists(full_path):
        api.upload_file(path_or_fileobj=full_path, path_in_repo=file, repo_id=repo_id)

print(f"HF Push Complete. Model available at: https://huggingface.co/{repo_id}")
