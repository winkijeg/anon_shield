import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from realm_testing.anon_shield.constants import (
    DATA_DIR,
    MODELS_DIR,
    SENTENCE_TRANSFORMER_MODEL,
)

sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

# Configuration
NUM_CASES_PER_CLASS = 1000
MAX_CONTEXT_LEN = 5
MAX_CONTEXT_TOKENS = 2 * MAX_CONTEXT_LEN  # MAX_CONTEXT_LEN left + MAX_CONTEXT_LEN right
DATA_FILE = DATA_DIR / f"test_cases_phone_{NUM_CASES_PER_CLASS}_{MAX_CONTEXT_LEN}.csv"


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_data():
    df = pd.read_csv(DATA_FILE)
    X = (df["left_context"] + " " + df["right_context"]).tolist()
    y = df["label"].tolist()
    return X, y


def get_transformer_embeddings(texts, tokenizer, model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_CONTEXT_TOKENS,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pooled = outputs.pooler_output  # shape: (batch_size, hidden_size)
            embeddings.append(pooled.cpu().numpy())
    # Concatenate all batches
    return np.vstack(embeddings)


def main():

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_folder = MODELS_DIR / run_timestamp
    run_folder.mkdir()

    print("Loading data...")
    X_texts, y = load_data()
    y = np.array(y)

    print("Generating embeddings...")
    X_embeddings = sentence_model.encode(X_texts, batch_size=32, show_progress_bar=True)
    print(f"Shape of embeddings: {X_embeddings.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: PCA + Classifier
    pca = PCA()
    classifiers = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "svm": SVC(class_weight="balanced", probability=True),
    }

    param_grid = {"pca__n_components": list(range(50, 101, 10))}  # 50 to 100

    for clf_name, clf in classifiers.items():
        print(f"Running GridSearchCV for {clf_name}...")

        pipeline = Pipeline([("pca", pca), ("clf", clf)])

        grid_search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=3, scoring="f1", verbose=2, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {clf_name}: {grid_search.best_params_}")
        print("Classification report on test set:")
        y_pred = grid_search.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Save the best PCA and classifier
        pca_model_file = run_folder / f"pca_model_{clf_name}.joblib"
        clf_model_file = run_folder / f"{clf_name}_model.joblib"

        best_pipeline = grid_search.best_estimator_
        joblib.dump(best_pipeline.named_steps["pca"], pca_model_file)
        joblib.dump(best_pipeline.named_steps["clf"], clf_model_file)

    # Save metadata
    training_metadata = {
        "num_samples_per_class": NUM_CASES_PER_CLASS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "timestamp": run_timestamp,
    }
    training_metadata_file = run_folder / f"meta_data.json"
    with open(training_metadata_file, "w") as f:
        json.dump(training_metadata, f, indent=2)

    manifest_file = MODELS_DIR / "manifest.json"
    manifest = {
        "runtime_models": run_timestamp,
        "classifier_type": "svm",
        "sentence_transformer_model": SENTENCE_TRANSFORMER_MODEL,
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Training complete. Models saved.")


if __name__ == "__main__":
    main()
