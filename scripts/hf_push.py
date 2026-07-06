from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

load_dotenv('/opt/ava/.env')
token = os.getenv("ACCESS_TOKEN")
repo_id = "artifact-virtual/raven-v1.1-sovereign"
local_folder = "/home/adam/worxpace/gladius/raven"

if not token:
    print("Error: No HF token found in .env")
    exit(1)

login(token=token)
api = HfApi()
api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

files_to_upload = {"weights/raven-v1-q8_0.gguf": "raven-v1-q8_0.gguf", "MODEL_CARD.md": "README.md", "cid.txt": "cid.txt"}
for local_path, repo_path in files_to_upload.items():
    full_path = os.path.join(local_folder, local_path)
    if os.path.exists(full_path):
        api.upload_file(path_or_fileobj=full_path, path_in_repo=repo_path, repo_id=repo_id)
