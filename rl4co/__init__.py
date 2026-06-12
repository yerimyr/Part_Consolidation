from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

try:
    __version__ = get_version(__package__ or "rl4co")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
