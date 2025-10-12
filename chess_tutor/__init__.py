"""Adaptive Chess Tutor package initializer."""
from .tutor_core import ChessTutor
from . import config

__all__ = [
    "ChessTutor",
    "config",
]

__version__ = "0.1.1"

def main():  # Allows: python -m chess_tutor
    from .tutor_cli import main as run
    run()
