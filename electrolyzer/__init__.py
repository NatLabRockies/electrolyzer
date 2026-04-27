"""Top-level package for Electrolyzer."""

__author__ = """Christopher Bay"""
__email__ = "christopher.bay@nrel.gov"
__version__ = "0.2.0"

from pathlib import Path


BERT_ROOT_DIR = Path(__file__).resolve().parent
BERT_EXAMPLE_DIR = BERT_ROOT_DIR.parent / "examples"
# BERT_LIBRARY_DIR = BERT_ROOT_DIR.parent / "library"
