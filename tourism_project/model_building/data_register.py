from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

HF_USERNAME = "karthikei94"
DATASET_REPO_NAME = "tourism-customer-purchase-prediction"
repo_id = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
repo_type = "dataset"

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is missing. Add it to your .env file for local runs or GitHub secrets for Actions.")

api = HfApi(token=hf_token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=hf_token)
    print(f"Dataset repo '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset folder uploaded successfully.")
