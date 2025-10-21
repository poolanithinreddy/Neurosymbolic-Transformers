import os


def test_fever_tsv_schema():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "fever.tsv")
    path = os.path.abspath(path)
    assert os.path.exists(path)
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header == ["split", "claim", "label", "evidence"]
        labels = set()
        n = 0
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            assert len(parts) == 4
            split, claim, label, evidence = parts
            assert split in {"train", "dev", "test"}
            assert isinstance(claim, str) and len(claim) > 0
            assert isinstance(evidence, str) and len(evidence) > 0
            labels.add(label)
            n += 1
        assert labels.issubset({"Supported", "Refuted", "NEI"})
        assert n > 0
