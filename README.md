# Wellness Tourism MLOps Prediction

This project builds an end-to-end MLOps workflow for predicting whether a customer is likely to purchase a Wellness Tourism Package. It includes data registration, data preparation, model experimentation with MLflow, model training, Hugging Face model/dataset upload, Streamlit deployment, and a GitHub Actions CI/CD workflow.

## Project Structure

```text
.
├── Learner_Template_Notebook_AML_and_MLOps_Project.ipynb
├── tourism.csv
├── tourism_project/
│   ├── data/
│   ├── model_building/
│   ├── deployment/
│   ├── hosting.py
│   └── requirements.txt
├── reference/
├── .env.example
└── .gitignore
```

## Environment Setup

Create and activate the Conda environment:

```bash
conda create -n tourism-mlops -c conda-forge python=3.11 pandas=2.2.2 scikit-learn=1.6.0 joblib notebook ipykernel -y
conda activate tourism-mlops
pip install xgboost==2.1.4 mlflow==3.0.1 pyngrok==7.2.12 huggingface_hub==0.32.6 datasets==3.6.0 streamlit==1.43.2 joblib==1.5.1 python-dotenv==1.1.1
python -m ipykernel install --name tourism-mlops --display-name "Python (tourism-mlops)"
```

Open the notebook:

```bash
jupyter notebook Learner_Template_Notebook_AML_and_MLOps_Project.ipynb
```

Select the kernel:

```text
Python (tourism-mlops)
```

## Local Secrets

Copy `.env.example` to `.env` and add your tokens:

```bash
cp .env.example .env
```

```text
HF_TOKEN=hf_your_actual_token
NGROK_AUTH_TOKEN=your_ngrok_token
```

The `.env` file is ignored by Git and should not be committed.

## GitHub Actions Secret

For CI/CD, add this repository secret in GitHub:

```text
HF_TOKEN
```

Path:

```text
Settings > Secrets and variables > Actions > New repository secret
```

## Workflow

Run the notebook cells in order. The notebook generates the project scripts used by the pipeline:

- `data_register.py`: uploads the raw dataset to Hugging Face Datasets
- `prep.py`: creates train/test splits and uploads them
- `train.py`: trains an XGBoost classifier, logs metrics with MLflow, and uploads the model
- `deployment/app.py`: Streamlit app for prediction
- `hosting.py`: uploads deployment files to Hugging Face Spaces
- GitHub Actions YAML: automates the end-to-end workflow

## Git Push

Recommended GitHub repository name:

```text
wellness-tourism-mlops-prediction
```

Push commands:

```bash
git add .gitignore .env.example README.md Learner_Template_Notebook_AML_and_MLOps_Project.ipynb tourism.csv tourism_project reference
git commit -m "Add wellness tourism MLOps project"
git branch -M main
git remote add origin https://github.com/<your-github-username>/wellness-tourism-mlops-prediction.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin https://github.com/<your-github-username>/wellness-tourism-mlops-prediction.git
git push -u origin main
```
