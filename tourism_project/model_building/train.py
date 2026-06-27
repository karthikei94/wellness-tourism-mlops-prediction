import os
from dotenv import load_dotenv
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HF_USERNAME = "karthikei94"
DATASET_REPO_NAME = "tourism-customer-purchase-prediction"
MODEL_REPO_NAME = "tourism-purchase-model"
DATASET_REPO_ID = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
MODEL_REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"
MODEL_FILENAME = "best_tourism_purchase_model_v1.joblib"

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-purchase-prediction")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is missing. Add it to your .env file for local runs or GitHub secrets for Actions.")

api = HfApi(token=hf_token)

Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/ytrain.csv").squeeze("columns")
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/ytest.csv").squeeze("columns")

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features),
)

scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 150],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__reg_lambda": [1, 5],
}

with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1",
    )
    grid_search.fit(Xtrain, ytrain)

    for i, params in enumerate(grid_search.cv_results_["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", grid_search.cv_results_["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", grid_search.cv_results_["std_test_score"][i])

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    threshold = 0.45
    train_proba = best_model.predict_proba(Xtrain)[:, 1]
    test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_train = (train_proba >= threshold).astype(int)
    y_pred_test = (test_proba >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    tn, fp, fn, tp = confusion_matrix(ytest, y_pred_test).ravel()
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
        "test_roc_auc": roc_auc_score(ytest, test_proba),
        "test_true_negative": tn,
        "test_false_positive": fp,
        "test_false_negative": fn,
        "test_true_positive": tp,
    })

    joblib.dump(best_model, MODEL_FILENAME)
    mlflow.log_artifact(MODEL_FILENAME, artifact_path="model")

    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"Model repo '{MODEL_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, token=hf_token)
        print(f"Model repo '{MODEL_REPO_ID}' created.")

    api.upload_file(
        path_or_fileobj=MODEL_FILENAME,
        path_in_repo=MODEL_FILENAME,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )

print(f"Model saved and uploaded as {MODEL_FILENAME}.")
