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
api.upload_file(path_or_fileobj=os.path.join(local_folder, ".gitattributes"), path_in_repo=".gitattributes", repo_id=repo_id)
