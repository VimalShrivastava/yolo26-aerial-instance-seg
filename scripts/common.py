"""Small shared helpers: every script takes its data/output location from the
command line instead of a hard-coded path.

* ``--root <dir>``   dataset root (layout described in the top-level README).
                    Falls back to the environment variable named by the script
                    (``ISAID_ROOT`` for iSAID, ``NWPU_ROOT`` for NWPU VHR-10).
* ``--config <yml>`` training settings for the Mask R-CNN scripts.

Both flags are removed from ``sys.argv`` before the script's own ``argparse``
runs, so they are accepted by every script without touching its own parser.
"""
import os
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _pop_flag(flag):
    """Remove ``flag value`` / ``flag=value`` from sys.argv and return the value."""
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
            del argv[i]
            return value
    return None


def resolve_root(env_var, flag="--root", required=True):
    """Return the dataset root given as ``flag`` or in environment variable ``env_var``."""
    value = _pop_flag(flag) or os.environ.get(env_var)
    if value:
        return Path(value).expanduser().resolve()
    if any(a in ("-h", "--help") for a in sys.argv):
        return Path(".")  # let argparse print its help text
    if required:
        sys.exit(f"error: pass {flag} <directory> or set the environment variable {env_var}")
    return Path(f"<{flag} not given>")


def load_config(default_name):
    """Load a YAML file from configs/ (override with ``--config <file>``)."""
    value = _pop_flag("--config")
    path = Path(value) if value else CONFIG_DIR / default_name
    with open(path) as f:
        return yaml.safe_load(f)
