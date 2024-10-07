import string


def preprocess_text(text, ignore_puncs=None):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        if ignore_puncs is not None and p in ignore_puncs:
            continue
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text

