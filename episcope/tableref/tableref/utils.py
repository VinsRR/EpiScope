import re
import string
import unicodedata

# Optional dependency for stopwords
try:
    import nltk
    # Attempt to access stopwords, and if it fails, download them
    try:
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))
    except OSError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords")
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))

except Exception:
    # fallback minimal stopword list if nltk not available
    print("NLTK not found, using a fallback stopword list.")
    _STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "to", "from", "up", "down", "out", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"
    }

# Regex helpers
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", flags=re.I)
AUTHOR_YEAR_RE = re.compile(r"([A-Z][A-Za-z\-]+(?: et al\.)?)\s*,?\s*(\d{4})")


def default_normalize(text: str) -> str:
    """Default text normalization function."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # lowercasing
    text = text.lower()
    # strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # remove stopwords (token-level)
    tokens = [tok for tok in text.split() if tok not in _STOPWORDS]
    return " ".join(tokens)

def simple_normalize(s: str, normalize_fn=default_normalize):
    """Wrapper to use the existing normalization function when available."""
    if s is None:
        return ""
    try:
        return normalize_fn(s)
    except Exception:
        # fallback simple ascii normalization
        s = unicodedata.normalize("NFKD", str(s))
        s = s.encode("ascii", "ignore").decode("ascii", "ignore")
        s = re.sub(r"[^\w\s]", " ", s).lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s
