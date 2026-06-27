from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os
from dotenv import load_dotenv

HF_USERNAME = "karthikei94"
SPACE_REPO_NAME = "wellness-tourism-purchase-prediction"
repo_id = f"{HF_USERNAME}/{SPACE_REPO_NAME}"

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is missing. Add it to your .env file for local runs or GitHub secrets for Actions.")

api = HfApi(token=hf_token)

try:
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        token=hf_token,
    )
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
)
print("Deployment files uploaded successfully.")
