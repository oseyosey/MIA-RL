"""
Dataset Paraphrase Module

This module provides functionality to paraphrase datasets using LLM APIs through LiteLLM.
It supports various LLM providers, with a focus on Google Gemini API.
"""

from .paraphrase import DatasetParaphraser
from .config import ParaphraseConfig

__all__ = ["DatasetParaphraser", "ParaphraseConfig"]
