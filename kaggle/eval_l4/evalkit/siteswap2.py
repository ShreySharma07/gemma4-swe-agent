"""
A12v validated site-swap + A5 keyword-free recompute (CPU, offline, no model); logic in necessity.py, pre-registered.
Run inside the venv: python -m evalkit.siteswap2
"""
from .necessity import main_siteswap_v2

if __name__ == "__main__":
    main_siteswap_v2()
