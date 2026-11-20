"""Per-model checkpoint epoch specification: parse --checkpoint-epoch values, resolve per model id.

Pure functions, no I/O. A repeatable ``--checkpoint-epoch`` value is either a bare ``N``
(the default epoch of every model, at most one) or ``<id>=N`` (the epoch of one model id,
split at the last '=' sign).
"""

from collections.abc import Iterable, Mapping, Sequence


def _parse_epoch(text: str, value: str) -> int:
    """Return text as a non-negative integer epoch.

    Args:
        text: the epoch text to convert.
        value: the whole checkpoint epoch value, named in the refusal.

    Returns:
        The epoch.

    Raises:
        ValueError: naming value, when text is not a non-negative integer.

    Complexity: O(len(text)).
    """
    try:
        epoch = int(text)
    except ValueError:
        epoch = -1
    if epoch < 0:
        raise ValueError(f"Checkpoint epoch {value!r}: the epoch must be a non-negative integer (N or <model id>=N)")
    return epoch


def _add_id_value(by_id: dict[str, int], value: str) -> None:
    """Add the '<id>=N' value to by_id.

    Args:
        by_id: the mapping of model id to epoch, extended in place.
        value: a checkpoint epoch value holding an '=' sign.

    Raises:
        ValueError: naming value or the model id, when the id is empty or repeated, or N is invalid.

    Complexity: O(len(value)).
    """
    model_id, _, text = value.rpartition("=")
    if not model_id:
        raise ValueError(f"Checkpoint epoch {value!r} has an empty model id before '='")
    epoch = _parse_epoch(text, value)
    if model_id in by_id:
        raise ValueError(f"Model id {model_id!r} is given twice in the checkpoint epoch values")
    by_id[model_id] = epoch


def parse_epoch_specs(values: Sequence[str], default: int) -> tuple[int, dict[str, int]]:
    """Parse the repeatable checkpoint epoch values.

    Args:
        values: the values, each a bare 'N' or '<id>=N'.
        default: the default epoch when no bare value is given, supplied by the caller.

    Returns:
        The default epoch and the mapping of model id to epoch.

    Raises:
        ValueError: naming the offending value or id, first fault in list order; two or more
            bare values are refused naming all of them.

    Complexity: O(n) in the total length of values.
    """
    by_id: dict[str, int] = {}
    bare: list[str] = []
    for value in values:
        if "=" in value:
            _add_id_value(by_id, value)
        else:
            _parse_epoch(value, value)
            bare.append(value)
    if len(bare) > 1:
        raise ValueError(f"Several default checkpoint epochs given: {', '.join(repr(v) for v in bare)}")
    result = (int(bare[0]) if bare else default, by_id)
    assert result[0] >= 0, "model_epoch_spec.parse_epoch_specs post: result[0] >= 0"
    assert len(result[1]) <= len(values), "model_epoch_spec.parse_epoch_specs post: len(result[1]) <= len(values)"
    return result


def resolve_model_epochs(default_epoch: int, by_id: Mapping[str, int], model_ids: Iterable[str]) -> dict[str, int]:
    """Give each model id its epoch.

    Args:
        default_epoch: the epoch of a model id absent from by_id.
        by_id: the per-id epochs.
        model_ids: the discovered model ids.

    Returns:
        The mapping of each model id to its epoch.

    Raises:
        ValueError: naming each id of by_id matching no model, and listing the known ids.

    Complexity: O(n log n) in the number of ids.
    """
    known = list(model_ids)
    unknown = sorted(set(by_id) - set(known))
    if unknown:
        listed = sorted({model_id for model_id in known if model_id})
        raise ValueError(f"Checkpoint epoch given for unknown model id(s) {unknown}; known model ids: {listed}")
    result = {model_id: by_id.get(model_id, default_epoch) for model_id in known}
    assert set(result) == set(known), "model_epoch_spec.resolve_model_epochs post: set(result) == set(model_ids)"
    return result
