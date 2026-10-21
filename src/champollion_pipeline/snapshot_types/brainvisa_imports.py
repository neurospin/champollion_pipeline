"""Guarded BrainVISA imports shared by the snapshot generators: the single patch point for tests."""

IMPORT_ERRORS: dict[str, str] = {}
try:
    import anatomist.headless as ana
except ImportError as exc:
    ana = None
    IMPORT_ERRORS["anatomist.headless"] = str(exc)
try:
    from cortical_tiles import config
except ImportError as exc:
    config = None
    IMPORT_ERRORS["cortical_tiles.config"] = str(exc)
try:
    from soma import aims
except ImportError as exc:
    aims = None
    IMPORT_ERRORS["soma.aims"] = str(exc)

_ATTRIBUTES = {"anatomist.headless": "ana", "cortical_tiles.config": "config", "soma.aims": "aims"}


def require(name):
    """Return the guarded module ``name`` as bound here now; raise ImportError naming it when absent."""
    module = globals()[_ATTRIBUTES[name]]
    if module is None:
        raise ImportError(f"{name} is not importable: {IMPORT_ERRORS.get(name, 'not installed')}")
    return module
