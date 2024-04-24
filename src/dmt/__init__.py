from importlib import metadata

__version__ = metadata.version(__name__)

from .libdmt import FDMT

__all__ = ["FDMT"]
