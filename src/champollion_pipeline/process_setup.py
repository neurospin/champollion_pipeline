"""Per-process logging setup for the pipeline entry points.

champollion_V1 and cortical_tiles configure the root logger when their logging
modules are imported (``logging.basicConfig`` then ``setFormatter`` on every root
handler). Those modules cannot be edited, so the pipeline imports them first,
removes the plain StreamHandler they added, then installs the unified
champollion_utils handlers and crash hook.
"""

import importlib
import logging

from champollion_utils import init_process

# Frozen-submodule modules that add a root StreamHandler when imported.
FROZEN_LOGGING_MODULES = ("champollion.utils.logs", "cortical_tiles.config.logs")


def detach_frozen_root_handlers() -> list[logging.Handler]:
    """Remove the plain root StreamHandlers added by the frozen submodules.

    Each module in FROZEN_LOGGING_MODULES is imported first (when installed) so
    its handler exists before removal. Only handlers whose type is exactly
    ``logging.StreamHandler`` are removed: the champollion_utils handlers, file
    handlers and pytest's capture handler are subclasses and stay. Handlers are
    removed, not closed. Safe to call more than once.

    Returns:
        The handlers removed from the root logger.
    """
    for module in FROZEN_LOGGING_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            # Not installed in this pixi environment (e.g. no V1 in brainvisa).
            pass
    root = logging.getLogger()
    removed = [h for h in root.handlers if type(h) is logging.StreamHandler]
    for handler in removed:
        root.removeHandler(handler)
    return removed


def init_pipeline_process() -> logging.Logger:
    """Detach the frozen submodules' root handlers, then run init_process().

    Returns:
        The root logger configured by champollion_utils.init_process().
    """
    detach_frozen_root_handlers()
    return init_process()
