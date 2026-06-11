#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime settings for the prompt experiments.

Prefer environment variables for public repositories. A local config.py file is
still supported for older workflows, but it is optional and ignored by git.
"""

import os


try:
    from config import (  # type: ignore
        API_DELAY as CONFIG_API_DELAY,
        API_KEY as CONFIG_API_KEY,
        FULL_TEST as CONFIG_FULL_TEST,
        MAX_TOKENS as CONFIG_MAX_TOKENS,
        MODEL as CONFIG_MODEL,
        TEMPERATURE as CONFIG_TEMPERATURE,
        TEST_SAMPLE_SIZE as CONFIG_TEST_SAMPLE_SIZE,
    )
except ImportError:
    CONFIG_API_KEY = None
    CONFIG_MODEL = "gpt-4o-mini"
    CONFIG_TEMPERATURE = 0.4
    CONFIG_MAX_TOKENS = 10
    CONFIG_TEST_SAMPLE_SIZE = 20
    CONFIG_FULL_TEST = False
    CONFIG_API_DELAY = 0.02


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


API_KEY = os.getenv("OPENAI_API_KEY") or CONFIG_API_KEY
MODEL = os.getenv("OPENAI_MODEL", CONFIG_MODEL)
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", CONFIG_TEMPERATURE))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", CONFIG_MAX_TOKENS))
TEST_SAMPLE_SIZE = int(os.getenv("TEST_SAMPLE_SIZE", CONFIG_TEST_SAMPLE_SIZE))
FULL_TEST = _env_bool("FULL_TEST", CONFIG_FULL_TEST)
API_DELAY = float(os.getenv("API_DELAY", CONFIG_API_DELAY))


def require_api_key():
    if not API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell or create a "
            "local config.py from config.py.example."
        )
    return API_KEY
