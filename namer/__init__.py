"""Генератор названий карточек товара для маркетплейсов (правила ПРОМПТ v5 + обучаемые веса)."""
from .dictionaries import Dictionaries
from .model import Model
from .generator import generate

__all__ = ["Dictionaries", "Model", "generate"]
__version__ = "0.1.0"
