"""
Unified sandbox backends for differential-expression benchmarking.

Functions:
    - get_backend: Return the requested sandbox backend

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

from .base_mean_wald import BaseMeanWaldBackend
from .tmm_lrt import TMMLRTBackend


def get_backend(name: str):
    """Return the requested sandbox backend."""
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in {"base_mean_wald", "pydiffexpress"}:
        return BaseMeanWaldBackend()
    if normalized in {"tmm_lrt"}:
        return TMMLRTBackend()
    raise ValueError(f"Unsupported backend: {name}")


__all__ = ["BaseMeanWaldBackend", "TMMLRTBackend", "get_backend"]
