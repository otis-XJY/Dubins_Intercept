"""Modular policy/critic network building blocks.

This package intentionally lives under ``marl/`` to avoid clashing with the
``marl/models.py`` module name.
"""

from .network import UAVInterceptionNetwork

__all__ = ["UAVInterceptionNetwork"]
