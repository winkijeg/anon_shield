import json

import joblib
from sentence_transformers import SentenceTransformer

from realm_testing.anon_shield.constants import (
    MASKING_STRING,
    MODELS_DIR,
    SENTENCE_TRANSFORMER_MODEL,
)
from realm_testing.anon_shield.regex_patterns import REGEX_PATTERNS

with open(MODELS_DIR / "manifest.json") as f:
    manifest = json.load(f)

CUR_MODEL_DIR = MODELS_DIR / manifest["runtime_models"]

with open(CUR_MODEL_DIR / "meta_data.json") as f:
    meta_data = json.load(f)

MAX_CONTEXT_TOKENS = meta_data["max_context_tokens"]

CLASSIFIER_TYPE = manifest["classifier_type"]
PCA_MODEL_FILE = CUR_MODEL_DIR / f"pca_model_{CLASSIFIER_TYPE}.joblib"
CLASSIFIER_MODEL_FILE = CUR_MODEL_DIR / f"{CLASSIFIER_TYPE}_model.joblib"


class PIIClassifier:
    def __init__(self):

        self.model = self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        self.pca = joblib.load(PCA_MODEL_FILE)
        self.clf = joblib.load(CLASSIFIER_MODEL_FILE)

    def find_potential_pii(self, text):
        """Identify potential PII using regex"""
        potential_pii = []
        for pii_type, pattern in REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                potential_pii.append(
                    {"pii_type": pii_type, "match": match.group(), "span": match.span()}
                )
        return potential_pii

    def extract_context(self, text, span):
        """Extract up to 5 tokens left and right of the detected span"""
        left_text = text[: span[0]].strip()
        right_text = text[span[1] :].strip()

        left_tokens = left_text.split()[-5:]
        right_tokens = right_text.split()[:5]

        context = " ".join(left_tokens + right_tokens)
        return context

    def embed_context(self, context_text):
        """Get embedding for context text"""
        embedding = self.model.encode(context_text)
        return embedding

    def classify_context(self, embedding):
        """Classify context as PII or not (1 or 0)"""
        reduced = self.pca.transform([embedding])
        prob_positive = self.clf.predict_proba(reduced)[0][1]  # probability for class 1
        pred = self.clf.predict(reduced)[0]
        return pred == 1, prob_positive

    def mask_pii(self, text, span):
        """Mask the PII occurrence in text"""
        start, end = span
        return text[:start] + MASKING_STRING + text[end:]

    def process_text(self, text):
        """Two-stage PII detection and masking"""
        potential_pii_list = self.find_potential_pii(text)
        final_text = text
        for pii in reversed(potential_pii_list):  # reversed to not mess up indices
            context = self.extract_context(final_text, pii["span"])
            embedding = self.embed_context(context)
            is_pii, prob_pii = self.classify_context(embedding)

            if is_pii:
                final_text = self.mask_pii(final_text, pii["span"])
        return final_text, prob_pii


if __name__ == "__main__":

    classifier = PIIClassifier()
    # example_prompt = "Please office at (123) 456-7890 as soon as possible."
    example_prompt = "dummy our is (123) 456-7890 call me."

    masked_text, prob_pii = classifier.process_text(example_prompt)

    print("Masked text:", masked_text, f"({prob_pii:.2f})")
