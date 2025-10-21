import os
import sys

# Ensure project root (nst/) is on sys.path so `import logic.*` works
THIS_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
