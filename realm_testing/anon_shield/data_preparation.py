# src/realm_testing/anon_shield/data_preparation.py

import csv
import random
from pathlib import Path

from faker import Faker

fake = Faker("en_US")

# Context tokens (max 5 left / 5 right tokens)
PERSONAL_CONTEXT_LEFT = ["I", "my", "our", "we", "us"]
PERSONAL_CONTEXT_RIGHT = ["call", "text", "reach", "message", "contact"]

NEUTRAL_CONTEXT_LEFT = ["table", "number", "field", "row", "record"]
NEUTRAL_CONTEXT_RIGHT = ["entry", "example", "sample", "test", "dummy"]

NUM_CASES_PER_CLASS = 200
MAX_CONTEXT_LEN = 5
OUTPUT_FILE = (
    Path(__file__).parent
    / "data"
    / f"test_cases_phone_{NUM_CASES_PER_CLASS}_{MAX_CONTEXT_LEN}.csv"
)


def generate_phone_number():
    return fake.phone_number()


def generate_context(context_list, num_tokens=MAX_CONTEXT_LEN):
    # Randomly pick context words (max 5)
    context_size = random.randint(1, num_tokens)
    return " ".join(random.choices(context_list, k=context_size))


def generate_case(is_positive=True):
    phone_number = generate_phone_number()
    if is_positive:
        left_context = generate_context(PERSONAL_CONTEXT_LEFT)
        right_context = generate_context(PERSONAL_CONTEXT_RIGHT)
        label = 1
    else:
        left_context = generate_context(NEUTRAL_CONTEXT_LEFT)
        right_context = generate_context(NEUTRAL_CONTEXT_RIGHT)
        label = 0

    # Concatenate as one string (LLM prompt style)
    prompt = f"{left_context} {phone_number} {right_context}"
    return {
        "prompt": prompt,
        "phone_number": phone_number,
        "left_context": left_context,
        "right_context": right_context,
        "label": label,
    }


def create_test_cases():
    cases = []
    for _ in range(NUM_CASES_PER_CLASS):
        cases.append(generate_case(is_positive=True))
    for _ in range(NUM_CASES_PER_CLASS):
        cases.append(generate_case(is_positive=False))

    # Shuffle the final dataset
    random.shuffle(cases)

    # Write to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "phone_number",
                "left_context",
                "right_context",
                "label",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(case)

    print(f"Generated {len(cases)} test cases in: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_test_cases()
