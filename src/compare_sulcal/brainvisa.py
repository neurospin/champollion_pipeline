"""Optional BrainVISA imports (PyAIMS, cortical_tiles subjects lookup) shared by the compare modes."""

try:
    from soma import aims
except ImportError as _aims_import_error:
    aims = None
    _AIMS_IMPORT_ERROR: "ImportError | None" = _aims_import_error
else:
    _AIMS_IMPORT_ERROR = None


try:
    from cortical_tiles.brainvisa.utils.subjects import get_all_subjects_as_dictionary
except ImportError as _cortical_tiles_import_error:
    get_all_subjects_as_dictionary = None
    _CORTICAL_TILES_IMPORT_ERROR: "ImportError | None" = _cortical_tiles_import_error
else:
    _CORTICAL_TILES_IMPORT_ERROR = None


_AIMS_UNAVAILABLE_MESSAGE = (
    "ERROR: PyAIMS (soma.aims) is not available in this environment; "
    "the masks, cortical_tiles, databases and crops subcommands require it (embeddings does not)."
)
