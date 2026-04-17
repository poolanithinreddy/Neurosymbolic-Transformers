"""NST model architectures."""

__all__ = ["NSTVeriModel", "build_fever_model"]


def __getattr__(name: str):
    if name == "NSTVeriModel":
        from models.nst_veri import NSTVeriModel
        return NSTVeriModel
    if name == "build_fever_model":
        from models.fever_nli import build_fever_model
        return build_fever_model
    raise AttributeError(f"module 'models' has no attribute {name!r}")
