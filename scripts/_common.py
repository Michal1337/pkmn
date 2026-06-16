"""Shared helpers for the local scripts."""

import importlib.util
import logging
import os
import sys

# kaggle_environments logs noisily about optional envs failing to import on
# `import`. Silence it before that happens.
logging.disable(logging.CRITICAL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")


def load_agent(path: str):
    """Import an agent module by file path and return its ``agent`` callable."""
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("submission_agent", path)
    module = importlib.util.module_from_spec(spec)
    # Make sibling files (deck.csv, helpers) resolvable from the agent dir.
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(module)
    return module.agent


def make_env():
    """Create the cabt environment (import deferred so logging is disabled)."""
    from kaggle_environments import make

    return make("cabt", configuration={})
