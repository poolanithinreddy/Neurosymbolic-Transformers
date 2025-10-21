from functools import lru_cache

try:
    import spacy  # type: ignore
except Exception:  # ImportError or runtime errors if not installed
    spacy = None  # type: ignore


@lru_cache(maxsize=1)
def _get_nlp():
    """
    Lazy-load spaCy model to avoid import-time failures when spaCy isn't installed.
    Requires the 'grounding' extra: pip install -e .[grounding]
    """
    if spacy is None:
        raise RuntimeError(
            "spaCy is not installed. Install NST with the 'grounding' extra: pip install -e .[grounding] and download the model: python -m spacy download en_core_web_sm"
        )
    try:
        return spacy.load("en_core_web_sm")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load spaCy model 'en_core_web_sm'. Please run: python -m spacy download en_core_web_sm"
        ) from exc


def load_aliases(path):
    aid2aliases, alias2id = {}, {}
    with open(path) as f:
        for line in f:
            qid, aliases = line.strip().split("\t")
            for a in aliases.split("|"):
                a_norm = a.lower()
                alias2id[a_norm] = qid
    return alias2id


def high_precision_link(text, alias2id):
    nlp = _get_nlp()
    doc = nlp(text)
    links = []
    for ent in doc.ents:
        key = ent.text.lower()
        if key in alias2id:
            links.append((ent.text, alias2id[key]))
    return links
