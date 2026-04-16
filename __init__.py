"""NST: Neuro-Symbolic Transformers.

Neural CEGIS — counterexample-guided training with augmented Lagrangian constraints.
GroundedVerifier — reusable verification layer for grounded AI systems.
"""

__all__ = ["__version__", "GroundedVerifier"]
__version__ = "0.2.0"


def __getattr__(name: str):
    if name == "GroundedVerifier":
        from nst.grounded_verifier import GroundedVerifier
        return GroundedVerifier
    raise AttributeError(f"module 'nst' has no attribute {name!r}")
