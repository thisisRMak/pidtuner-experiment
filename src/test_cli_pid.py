"""Unit tests for cli_pid.py's -p/-L/-m short-flag aliases.

Run with:
    python test_cli_pid.py
or:
    python -m unittest test_cli_pid -v

Invokes the script as a real subprocess (matching how a user would call it)
to confirm the short flags are equivalent to the pre-existing long flags.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

_CLI_PATH = os.path.join(os.path.dirname(__file__), "cli_pid.py")


def _run(args):
    result = subprocess.run(
        [sys.executable, _CLI_PATH, *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


class TestShortFlagAliases(unittest.TestCase):
    def test_short_flags_match_long_flags(self):
        long_out = _run(["--plant", "1/(90s+1)", "--L", "13",
                          "--method", "cohen_coon", "--json"])
        short_out = _run(["-p", "1/(90s+1)", "-L", "13",
                           "-m", "cohen_coon", "--json"])
        self.assertEqual(json.loads(long_out), json.loads(short_out))


if __name__ == "__main__":
    unittest.main()
