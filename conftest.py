"""Offline tests use APP_ENV=testing so committed overlays do not leak into settings."""

from __future__ import annotations

import os

os.environ.setdefault("APP_ENV", "testing")
