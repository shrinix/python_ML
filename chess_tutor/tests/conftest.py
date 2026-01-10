"""Test configuration: ensure imports work regardless of CWD.

- Put the workspace parent of `chess_tutor` on sys.path so `import chess_tutor.*` works.
- Avoid inserting the `chess_tutor` directory itself at position 0, which would
    break `import chess_tutor.tutor_core` by shadowing the package base.
"""
import sys, pathlib

THIS_DIR = pathlib.Path(__file__).resolve()
CHESS_TUTOR_DIR = THIS_DIR.parents[1]
WS_ROOT = THIS_DIR.parents[2]

# Ensure workspace root for `import chess_tutor.*`
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

# Also ensure the chess_tutor folder is on sys.path (not at position 0) so `import backend.*` works
if str(CHESS_TUTOR_DIR) not in sys.path:
    sys.path.append(str(CHESS_TUTOR_DIR))

