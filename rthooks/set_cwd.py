# rthooks/set_cwd.py — PyInstaller runtime hook: фиксируем cwd = папка с exe
import os, sys
from pathlib import Path

def _fix_cwd():
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        os.chdir(exe_dir)

_fix_cwd()
