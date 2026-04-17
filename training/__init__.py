"""NST training loops."""

__all__ = ["train_fever_veri", "train_fever_nst"]


def __getattr__(name: str):
    if name == "train_fever_veri":
        from training.train_fever_veri import train_fever_veri
        return train_fever_veri
    if name == "train_fever_nst":
        from training.train_fever_nst import train_fever_nst
        return train_fever_nst
    raise AttributeError(f"module 'training' has no attribute {name!r}")
