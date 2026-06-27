import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_USERNAME = "karthikei94"
DATASET_REPO_NAME = "tourism-customer-purchase-prediction"
repo_id = f"{HF_USERNAME}/{DATASET_REPO_NAME}"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is missing. Add it to your .env file for local runs or GitHub secrets for Actions.")

api = HfApi(token=hf_token)
DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop index/identifier fields that should not influence customer purchase prediction.
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Normalize inconsistent category spelling found in the dataset.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

print("Train/test split files uploaded successfully.")
