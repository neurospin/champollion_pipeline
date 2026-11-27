"""Per-subject sulcal voxel counts and mask statistics of the compare databases mode."""

import glob
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from os.path import join

import numpy as np

from champollion_pipeline.utils.graph_item import GraphItem
from compare_sulcal import brainvisa

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Campaign graph choice over discovered graphs (pure)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CampaignGraph:
    """The graph a campaign reads for one side and subject, and how many graphs of its label matched there."""

    side: str
    subject: str
    path: str
    matches: int


@dataclass(frozen=True)
class CampaignGraphs:
    """The graphs of one campaign chosen from graph discovery, and the (side, subject) pairs left without one."""

    label: str
    subjects: tuple[str, ...]
    graphs: tuple[CampaignGraph, ...]
    missing: tuple[tuple[str, str], ...]


def select_campaign_graphs(items: Sequence[GraphItem], label: str, sides: Sequence[str]) -> CampaignGraphs:
    """Choose a campaign's graphs: labelling set == label, hemisphere in sides, first per (side, subject).

    items are discover_graphs output, in natural path order. Each kept group of one side and subject gives one
    CampaignGraph: its first item, with matches = the group size. subjects lists every discovered subject (any
    labelling set or none) in first-seen order; missing lists, per side, the subjects without a graph of label.
    No I/O, inputs never mutated. Complexity O(len(items)).
    """
    subjects = tuple(dict.fromkeys(item.attributes["subject"] for item in items))
    groups: dict[tuple[str, str], list[GraphItem]] = {}
    for item in items:
        side = item.attributes["hemisphere"]
        if item.attributes["labelling_set"] == label and side is not None and side in sides:
            groups.setdefault((side, item.attributes["subject"]), []).append(item)
    graphs = tuple(
        CampaignGraph(side=side, subject=subject, path=group[0].path, matches=len(group))
        for side in sides
        for (group_side, subject), group in groups.items()
        if group_side == side
    )
    missing = tuple((side, subject) for side in sides for subject in subjects if (side, subject) not in groups)
    result = CampaignGraphs(label=label, subjects=subjects, graphs=graphs, missing=missing)
    assert len({(g.side, g.subject) for g in graphs}) == len(graphs), (
        "select_campaign_graphs post: one CampaignGraph per (side, subject)"
    )
    assert all(g.subject in subjects for g in graphs), "select_campaign_graphs post: every graph subject in subjects"
    return result


def list_labelling_sets(items: Sequence[GraphItem]) -> list[str]:
    """Sorted distinct labelling sets of the discovered graphs, None left out."""
    return sorted({item.attributes["labelling_set"] for item in items} - {None})


# --------------------------------------------------------------------------- #
# Database-mode module-level workers (must be picklable for joblib)
# --------------------------------------------------------------------------- #


def _get_subject_voxel_counts(sub):
    """Worker: load one graph and return (subject_name, {sulcus: voxel_count})."""
    matches = sorted(glob.glob(join(sub["dir"], sub["graph_file"])))
    if not matches:
        return sub["subject"], None
    if len(matches) > 1:
        logger.warning(f"{len(matches)} graphs match subject {sub['subject']}; using {matches[0]}")

    graph = brainvisa.aims.read(matches[0])
    return sub["subject"], _count_graph_voxels(graph)


def _count_graph_voxels(graph) -> dict[str, int]:
    """Return {sulcus: voxel_count} of a read graph, summed over the aims_ss, aims_bottom and aims_other buckets."""
    counts: dict[str, int] = {}
    for vertex in graph.vertices():
        name = vertex.get("name")
        if name is None:
            continue
        n = 0
        for bucket_name in ("aims_ss", "aims_bottom", "aims_other"):
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                n += len(list(bucket[0].keys()))
        counts[name] = counts.get(name, 0) + n
    return counts


def _get_graph_voxel_counts(subject: str, graph_path: str) -> tuple[str, dict]:
    """Worker: read the graph at graph_path and return (subject, {sulcus: voxel_count})."""
    return subject, _count_graph_voxels(brainvisa.aims.read(graph_path))


def _mask_stats(mask_dir: str) -> dict:
    """Return {relative_path: (max, sum, nonzero)} for all masks in mask_dir."""
    stats = {}
    for path in sorted(glob.glob(join(mask_dir, "*/*.nii.gz"))):
        rel = os.path.relpath(path, mask_dir)
        arr = np.asarray(brainvisa.aims.read(path))
        stats[rel] = (int(arr.max()), int(arr.sum()), int(np.count_nonzero(arr)))
    return stats
