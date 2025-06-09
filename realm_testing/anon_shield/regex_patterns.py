import re

# --- US phone numbers ---
# Example: (123) 456-7890, 123-456-7890, +1 123 456 7890, 123.456.7890, etc.
US_PHONE_REGEX = re.compile(
    r"""
    (?<!\d)                    # no digit before
    (?:\+1\s*)?                # optional +1 country code
    (?:\(\d{3}\)|\d{3})        # area code with or without ()
    [\s\-.]?                   # separator
    \d{3}                      # first 3 digits
    [\s\-.]?                   # separator
    \d{4}                      # last 4 digits
    (?!\d)                     # no digit after
    """,
    re.VERBOSE,
)

# Future: add more regex for SSN, credit cards, etc.
REGEX_PATTERNS = {
    "US_PHONE": US_PHONE_REGEX,
    # "SSN": re.compile(...),
    # "CREDIT_CARD": re.compile(...)
}
