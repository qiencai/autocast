"""Utility functions for AutoCast scripts."""

from pathlib import Path


def get_default_config_path() -> str:
    """Find the configs directory by searching upward for project root.

    Searches upward from this file for pyproject.toml (project root marker),
    then returns the path to the configs directory.

    Returns
    -------
    str
        Absolute path to the configs directory.

    Raises
    ------
    FileNotFoundError
        If project root (pyproject.toml) cannot be found.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:  # Stop at filesystem root
        if (current / "pyproject.toml").exists():
            config_dir = current / "configs"
            if not config_dir.exists():
                msg = f"Project root found at {current}, but configs directory missing"
                raise FileNotFoundError(msg)
            return str(config_dir)
        current = current.parent

    msg = "Could not find project root (pyproject.toml)"
    raise FileNotFoundError(msg)
