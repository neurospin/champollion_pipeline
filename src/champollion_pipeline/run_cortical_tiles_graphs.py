"""
Stage 2 graph handoff: the graphs of each subject folder, found by graph discovery, as the map cortical_tiles reads.

Without --path_to_graph, run_cortical_tiles lists the graphs below its input once (the StageIndexer port's
discover_graphs), chooses per subject folder and hemisphere the graph to crop (cortical_tiles' select_graphs: a graph
of the chosen labelling set in preference to an unlabelled one, then the first in natural path order, for now) and
hands generate_sulcal_regions the result as a ``--graphs`` file: {subject folder name: {"L": [.arg], "R": [.arg]}}.
The pipeline owns this choice; cortical_tiles does none once it is given a map (anticorruption layer: select_graphs
is its pure function, called through this module only).

Inputs:
    input_abs: absolute folder of subject folders; label: the --graph_label value or None
    discover: the port's discover_graphs (FileIndexStageIndexer("").discover_graphs)
Outputs:
    GraphListing: the map, the subject folders listed with an unlabelled graph, the subject folders with several
    labelling sets (to stop on, when no label was given); a graphs file for the crop command.
"""

import json
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory

from cortical_tiles.brainvisa.subject_graphs import SeveralLabellingSetsError, select_graphs

from champollion_pipeline.utils.graph_item import GraphItem

logger = logging.getLogger(
    "champollion_pipeline.run_cortical_tiles"
)  # the stage 2 logger, as the other stage 2 modules

GRAPH_SIDES = ("L", "R")
GRAPHS_FILENAME = "graphs.json"
GraphsMap = dict[str, dict[str, list[str]]]  # subject folder name -> side -> absolute .arg paths


@dataclass
class GraphListing:
    """What graph discovery chose below an input.

    graphs: subject folder name -> side -> the listed graphs (at most one per side, an empty list for a side without).
    unlabelled: subject folder name -> the sides whose listed graph is unlabelled.
    several_sets: subject folder path -> sorted labelling sets, for each subject folder holding several of them
        while no label was given.
    """

    graphs: GraphsMap = field(default_factory=dict)
    unlabelled: dict[str, tuple[str, ...]] = field(default_factory=dict)
    several_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def has_side(self, subject: str, side: str) -> bool:
        """True when a graph of ``side`` is listed for the subject folder ``subject``."""
        return bool(self.graphs.get(subject, {}).get(side))


def list_subject_graphs(input_abs: str, label: str | None, discover: Callable[[str], list[GraphItem]]) -> GraphListing:
    """List the graphs of each subject folder below ``input_abs``.

    A subject folder with no graph gets no entry. With ``label`` None and a subject folder holding graphs of two or
    more labelling sets, nothing is listed for it and it goes to ``several_sets``; all such folders are collected.

    Raises:
        LooseGraphError: from ``discover``, a graph directly in ``input_abs`` outside any subject folder.
        NotADirectoryError: from ``discover``, ``input_abs`` is no folder.

    Complexity: O(graphs below input_abs) plus the sort of each subject's graphs.
    """
    by_subject: dict[str, list[str]] = {}
    for item in discover(input_abs):
        by_subject.setdefault(str(item.attributes["subject"]), []).append(
            os.path.join(input_abs, *item.path.split("/"))
        )
    listing = GraphListing()
    for subject, arg_paths in by_subject.items():
        subject_dir = os.path.join(input_abs, subject)
        try:
            selection = select_graphs(subject_dir, arg_paths, label)
        except SeveralLabellingSetsError as error:
            listing.several_sets.update(error.sets_by_folder)
            continue
        listing.graphs[subject] = {side: list(selection.graphs[side]) for side in GRAPH_SIDES}
        if selection.unlabelled_sides:
            listing.unlabelled[subject] = tuple(selection.unlabelled_sides)
    return listing


def log_unlabelled(listing: GraphListing) -> None:
    """One WARNING per subject folder listed with an unlabelled graph, naming its hemispheres."""
    for subject, sides in sorted(listing.unlabelled.items()):
        logger.warning(f"{subject}: the listed graph of hemisphere {', '.join(sides)} is unlabelled")


def compute_several_sets_message(listing: GraphListing) -> str:
    """The ERROR text naming each subject folder with several labelling sets and its sets, and the option to give."""
    return str(SeveralLabellingSetsError(listing.several_sets))


@contextmanager
def write_graphs_file(graphs: GraphsMap) -> Iterator[str]:
    """Write ``graphs`` as JSON in a temporary folder and yield the file's path; the folder goes away on exit."""
    with TemporaryDirectory(prefix="graphs_map_") as folder:
        path = os.path.join(folder, GRAPHS_FILENAME)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(graphs, stream, indent=1, sort_keys=True)
        yield path
