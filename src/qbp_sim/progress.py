from __future__ import annotations

import os
import sys
from typing import TextIO


def should_use_progress(stream: TextIO | None = None) -> bool:
    target = sys.stderr if stream is None else stream
    if os.environ.get("CI"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    is_tty = getattr(target, "isatty", None)
    return bool(is_tty and is_tty())
