from pathlib import Path

# --- Transformer model ---
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# --- Context settings ---
MAX_CONTEXT_LEN = 5
MAX_CONTEXT_TOKENS = 2 * MAX_CONTEXT_LEN  # MAX_CONTEXT_LEN left + MAX_CONTEXT_LEN right

# --- Directories ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# --- Masking string ---
MASKING_STRING = "[MASKED]"
