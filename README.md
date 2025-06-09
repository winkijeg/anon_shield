# Anon Shield

A two-stage PII detection pipeline for any text blobs, i.e. large language model (LLM) prompts.

## Overview

**Anon Shield** scans LLM prompts for potential US phone numbers using regex patterns and then classifies the surrounding context (max. 5 tokens left & max. 5 tokens right) using a transformer-based embedding model, PCA, and one of two classifiers (Logistic Regression and SVM).

## Features

- **Stage 1:** Identify potential US phone number PII using regex.
- **Stage 2:** Classify left/right context as positive (actual PII) or negative (false match).
- **Masking:** Replace positively identified PII with `[MASKED]`.
- **Extensible:** Add new PII regex patterns (SSNs, credit cards, emails, etc.) easily.

## Project Structure

```
src/realm_testing/anon_shield/
├── data_preparation.py  # Create test cases
├── model_training.py    # Train PCA + classifier
├── pii_classifier.py    # Final detection and masking
├── regex_patterns.py    # Regex patterns for US phone numbers (and more)
├── models/              # Trained models and metadata (per run)
├── manifest.json        # Points to the current active run
└── data/                # Generated test cases
```

## Installation

1.  Create a new environment:
   ```
   conda create -n anon_shield_env python=3.10
   conda activate anon_shield
   ```

2.  Install dependencies:
   ```
   pip install sentence-transformers pandas scikit-learn
   ```

## Usage

1.	Generate test data:
   ```
   python src/realm_testing/anon_shield/data_preparation.py
   ```

2.	Train the models:
   ```
   python src/realm_testing/anon_shield/model_training.py
   ```

3.	Classify and mask PII in text:
   ```
   python src/realm_testing/anon_shield/pii_classifier.py
   ```

## License

MIT License