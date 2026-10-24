"""
A12 site-swap + A5 typology (CPU, offline, no model); the logic lives in necessity.py (pre-registered there).
Run inside the venv: python -m evalkit.siteswap
"""
from .necessity import main_siteswap

if __name__ == "__main__":
    main_siteswap()
