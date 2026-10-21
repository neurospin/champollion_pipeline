"""Embedding comparison (CKA, k-NN overlap, Procrustes, UMAP) of the compare embeddings mode."""

import csv
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
from matplotlib.figure import Figure
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------- #
# Embeddings-mode constants
# --------------------------------------------------------------------------- #

_EMBEDDINGS_SUFFIX = "_embeddings.csv"


_HEMI_BY_SIDE = {"L": "left", "R": "right"}


_EMBEDDINGS_ID_COLUMN = "ID"


_EMB_UMAP_COLUMNS = ["umap_a_x", "umap_a_y", "umap_b_x", "umap_b_y", "displacement"]


_EMB_COVARIATE_COLUMN = "covariate"


_MIN_SPEARMAN_PAIRS = 3


# --------------------------------------------------------------------------- #
# Embeddings-mode public helpers (pinned by tests)
# --------------------------------------------------------------------------- #


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centred linear CKA (Kornblith 2019), rows paired.

    Feature-space form: ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F).
    Returns nan when the denominator is 0 (constant matrix). O(n d^2), never
    builds an n x n Gram matrix.
    """
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    num = float(np.linalg.norm(yc.T @ xc, "fro") ** 2)
    denom = float(np.linalg.norm(xc.T @ xc, "fro") * np.linalg.norm(yc.T @ yc, "fro"))
    if denom == 0.0:
        return float("nan")
    return num / denom


def _drop_self_neighbour(indices: np.ndarray) -> np.ndarray:
    """Remove each row's own index from (n, k+1) kNN indices, giving (n, k).

    Rows where self is absent (duplicate points) drop their last column instead.
    """
    n, width = indices.shape
    is_self = indices == np.arange(n)[:, None]
    no_self = ~is_self.any(axis=1)
    is_self[no_self, width - 1] = True
    return indices[~is_self].reshape(n, width - 1)


def knn_overlap(emb_a: np.ndarray, emb_b: np.ndarray, k: int) -> np.ndarray:
    """Shape (n,): |N_k^A(i) & N_k^B(i)| / k, Euclidean, self excluded.

    Requires n > k.
    """
    nn_a = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb_a)
    nn_b = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb_b)
    idx_a = _drop_self_neighbour(nn_a.kneighbors(emb_a, return_distance=False))
    idx_b = _drop_self_neighbour(nn_b.kneighbors(emb_b, return_distance=False))
    n = emb_a.shape[0]
    result = np.empty(n, dtype=float)
    for i in range(n):
        result[i] = len(set(idx_a[i].tolist()) & set(idx_b[i].tolist())) / k
    return result


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Similarity Procrustes, reflection allowed.

    Centre both, find R via orthogonal_procrustes(src_c, tgt_c);
    scale = sv_sum / ||src_c||_F^2; return src_c @ R * scale + tgt_mean.
    If ||src_c|| == 0, every row equals tgt_mean.
    """
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    norm_sq = float(np.linalg.norm(src_c, "fro") ** 2)
    if norm_sq == 0.0:
        return np.tile(tgt_mean, (src.shape[0], 1))
    r_mat, sv_sum = orthogonal_procrustes(src_c, tgt_c)
    scale = sv_sum / norm_sq
    return src_c @ r_mat * scale + tgt_mean


def load_umap_model(path: str):
    """Load a UMAP model pickle with joblib.

    Looked up by name at call time (module global), never bound as a default
    arg/alias, so tests can monkeypatch it.
    """
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Embeddings-mode private helpers
# --------------------------------------------------------------------------- #


def _load_embeddings_csv(path: Path) -> "tuple[list[str], np.ndarray] | str":
    """Load an embeddings CSV (ID column + float feature columns).

    Returns (ids, matrix) on success, or a skip-reason string on error:
    'no_id_column', 'invalid_embeddings_csv', 'duplicate_subject_ids'.
    """
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or _EMBEDDINGS_ID_COLUMN not in reader.fieldnames:
                return "no_id_column"
            feature_cols = [c for c in reader.fieldnames if c != _EMBEDDINGS_ID_COLUMN]
            ids: list[str] = []
            rows_data: list[list[float]] = []
            for row in reader:
                subject = row[_EMBEDDINGS_ID_COLUMN]
                try:
                    values = [float(row[c]) for c in feature_cols]
                except (ValueError, TypeError):
                    return "invalid_embeddings_csv"
                ids.append(subject)
                rows_data.append(values)
    except OSError:
        return "missing_file"
    seen: set[str] = set()
    for s in ids:
        if s in seen:
            return "duplicate_subject_ids"
        seen.add(s)
    return ids, np.array(rows_data, dtype=np.float64)


def _list_embedding_pairs(set_a: Path, set_b: Path, regions: "list[str] | None", sides: list) -> list:
    """Return [(region, side), ...] pairs to compare.

    Default (regions is None): files present in both sets per side, sorted by
    region then L, R. Explicit regions: regions x sides as given.
    """
    if regions is not None:
        return [(r, s) for r in regions for s in sides]
    pairs: list[tuple[str, str]] = []
    for side in sides:
        hemi = _HEMI_BY_SIDE[side]
        suffix = f"_{hemi}{_EMBEDDINGS_SUFFIX}"
        names_a = {p.name[: -len(suffix)] for p in set_a.glob(f"*{suffix}")}
        names_b = {p.name[: -len(suffix)] for p in set_b.glob(f"*{suffix}")}
        for region in sorted(names_a & names_b):
            pairs.append((region, side))
    return pairs


def _compute_id_pairing(ids_a: list, ids_b: list) -> "tuple[list[int], list[int], int, int]":
    """Pair IDs from A and B; return (idx_a, idx_b, only_in_a, only_in_b).

    Paired indices are in set-A order.
    """
    set_b = {s: i for i, s in enumerate(ids_b)}
    idx_a: list[int] = []
    idx_b: list[int] = []
    for i, s in enumerate(ids_a):
        if s in set_b:
            idx_a.append(i)
            idx_b.append(set_b[s])
    only_in_a = len(ids_a) - len(idx_a)
    only_in_b = len(ids_b) - len(idx_b)
    return idx_a, idx_b, only_in_a, only_in_b


def _compute_umap_projection(
    umap_a_dir: "str | None",
    umap_b_dir: "str | None",
    region: str,
    side: str,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray] | str":
    """Project embeddings through UMAP models and align B onto A.

    Returns (umap_a_coords, umap_b_aligned) or a skip-reason string.
    Checks A first: 'missing_umap_a', 'missing_umap_b'.
    """
    hemi = _HEMI_BY_SIDE[side]
    pkl_name = f"umap_{region}_{hemi}.pkl"
    path_a = Path(umap_a_dir) / pkl_name
    path_b = Path(umap_b_dir) / pkl_name
    if not path_a.exists():
        return "missing_umap_a"
    if not path_b.exists():
        return "missing_umap_b"
    try:
        model_a = load_umap_model(str(path_a))
        umap_a = np.asarray(model_a.transform(emb_a.astype(float)), dtype=float)
    except Exception as exc:
        print(f"WARNING: umap_a transform failed for {region}/{side}: {exc}")
        return "umap_transform_failed_a"
    try:
        model_b = load_umap_model(str(path_b))
        umap_b_raw = np.asarray(model_b.transform(emb_b.astype(float)), dtype=float)
    except Exception as exc:
        print(f"WARNING: umap_b transform failed for {region}/{side}: {exc}")
        return "umap_transform_failed_b"
    umap_b_aligned = procrustes_align(umap_b_raw, umap_a)
    return umap_a, umap_b_aligned


def _compute_spearman(x: list, y: list) -> "float | None":
    """Spearman r of finite pairs; None if fewer than _MIN_SPEARMAN_PAIRS pairs."""
    pairs = [(xi, yi) for xi, yi in zip(x, y) if math.isfinite(xi) and math.isfinite(yi)]
    if len(pairs) < _MIN_SPEARMAN_PAIRS:
        return None
    xs, ys = zip(*pairs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat = spearmanr(xs, ys).statistic
    if math.isnan(stat):
        return None
    return float(stat)


def _load_covariate_csv(path: str, column: str) -> "dict[str, float] | str":
    """Load a covariate CSV into {subject: float}.

    Returns an error string on missing columns or duplicate subjects.
    Empty / non-numeric cells are silently skipped.
    """
    result: dict[str, float] = {}
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return f"covariate CSV has no columns: {path}"
            if "subject" not in reader.fieldnames:
                return f"covariate CSV missing 'subject' column: {path}"
            if column not in reader.fieldnames:
                return f"covariate CSV missing '{column}' column: {path}"
            for row in reader:
                subject = row["subject"]
                if subject in result:
                    return f"duplicate subject in covariate CSV: {subject}"
                val = row[column]
                if val is None or val.strip() == "":
                    continue
                try:
                    result[subject] = float(val)
                except (ValueError, TypeError):
                    pass
    except OSError as exc:
        return str(exc)
    return result


def _summarise_embeddings_region(
    rows: list,
    only_in_a: int,
    only_in_b: int,
    cka: float,
    n_features_a: int,
    n_features_b: int,
    umap_mode: bool,
    umap_coords: "tuple | None",
    subjects: list,
    covariate_map: "dict | None",
) -> dict:
    """Build the summary dict for one region/side."""
    knn_vals = [float(r["knn_overlap"]) for r in rows]
    entry: dict = {
        "n_subjects": len(rows),
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "n_features_a": n_features_a,
        "n_features_b": n_features_b,
        "cka": cka,
        "knn_overlap_mean": float(np.mean(knn_vals)),
        "knn_overlap_median": float(np.median(knn_vals)),
    }
    has_umap = umap_mode and umap_coords is not None
    displacements = [float(r["displacement"]) for r in rows] if has_umap else []
    if has_umap:
        entry["displacement_mean"] = float(np.mean(displacements))
        entry["displacement_median"] = float(np.median(displacements))
    if covariate_map is not None:
        cov_vals = [covariate_map.get(s, float("nan")) for s in subjects]
        entry["n_covariate"] = sum(1 for v in cov_vals if math.isfinite(v))
        entry["spearman_covariate_knn_change"] = _compute_spearman(cov_vals, [1.0 - v for v in knn_vals])
        if has_umap:
            entry["spearman_covariate_displacement"] = _compute_spearman(cov_vals, displacements)
        elif umap_mode:
            entry["spearman_covariate_displacement"] = None
    return entry


def _save_umap_comparison_png(
    out_path: Path,
    region: str,
    side: str,
    umap_a: np.ndarray,
    umap_b: np.ndarray,
    covariate_values: "list | None",
    column: "str | None",
) -> Path:
    """Save a side-by-side PNG comparing UMAP projections A and B."""
    fig = Figure(figsize=(12, 5.5))
    axes = fig.subplots(1, 2)
    all_x = np.concatenate([umap_a[:, 0], umap_b[:, 0]])
    all_y = np.concatenate([umap_a[:, 1], umap_b[:, 1]])
    x_lim = (float(all_x.min()), float(all_x.max()))
    y_lim = (float(all_y.min()), float(all_y.max()))

    if covariate_values is not None and any(v is not None for v in covariate_values):
        finite_vals = [v for v in covariate_values if v is not None]
        vmin = min(finite_vals)
        vmax = max(finite_vals)
        colors_valid = np.array([v if v is not None else float("nan") for v in covariate_values])
        has_cov = np.array([v is not None for v in covariate_values])
        for ax, coords, title in zip(
            axes, [umap_a, umap_b], ["set A (--umap_a)", "set B (--umap_b), Procrustes-aligned to A"]
        ):
            if has_cov.any():
                sc = ax.scatter(
                    coords[has_cov, 0],
                    coords[has_cov, 1],
                    c=colors_valid[has_cov],
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    s=8,
                    rasterized=True,
                )
            if (~has_cov).any():
                ax.scatter(
                    coords[~has_cov, 0],
                    coords[~has_cov, 1],
                    c="grey",
                    s=8,
                    rasterized=True,
                )
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_title(title)
        if has_cov.any():
            fig.colorbar(sc, ax=axes.tolist(), label=column or "covariate")
    else:
        for ax, coords, title in zip(
            axes, [umap_a, umap_b], ["set A (--umap_a)", "set B (--umap_b), Procrustes-aligned to A"]
        ):
            ax.scatter(coords[:, 0], coords[:, 1], s=8, rasterized=True)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_title(title)

    fig.suptitle(
        f"{region} {side}: separate UMAP fits, B aligned to A by Procrustes "
        f"(translation, rotation/reflection, isotropic scale)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    return out_path


def _normalize_json_value(value):
    """Recursively replace float nan/inf with None for strict JSON serialisation."""
    if isinstance(value, dict):
        return {k: _normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _skipped_embeddings_result(region: str, side: str, reason: str) -> dict:
    """Worker result for a region/side that could not be compared."""
    return {
        "key": f"{region}/{side}",
        "region": region,
        "side": side,
        "rows": [],
        "summary": None,
        "skipped": {"region": region, "side": side, "reason": reason},
        "umap_skipped": [],
    }


def _compare_embeddings_region_side(
    set_a: Path,
    set_b: Path,
    region: str,
    side: str,
    options: dict,
) -> dict:
    """Worker: compare one region/side pair of embeddings.

    options keys: k, umap_a, umap_b, covariate (dict|None), covariate_column, output, explicit.
    Returns dict with keys: key, region, side, rows, summary, skipped, umap_skipped.
    """
    key = f"{region}/{side}"
    hemi = _HEMI_BY_SIDE[side]
    path_a = set_a / f"{region}_{hemi}{_EMBEDDINGS_SUFFIX}"
    path_b = set_b / f"{region}_{hemi}{_EMBEDDINGS_SUFFIX}"
    explicit = options.get("explicit", False)

    # ── Load CSV files ─────────────────────────────────────────────────────
    result_a = _load_embeddings_csv(path_a)
    result_b = _load_embeddings_csv(path_b)
    if isinstance(result_a, str) or isinstance(result_b, str):
        if isinstance(result_a, str) and isinstance(result_b, str):
            reason = "missing_in_a_and_b" if explicit else result_a
        elif isinstance(result_a, str):
            reason = "missing_in_a"
        else:
            reason = "missing_in_b"
        return _skipped_embeddings_result(region, side, reason)
    ids_a, emb_a = result_a
    ids_b, emb_b = result_b

    # ── Pair by ID ─────────────────────────────────────────────────────────
    idx_a, idx_b, only_in_a, only_in_b = _compute_id_pairing(ids_a, ids_b)
    if len(idx_a) == 0:
        return _skipped_embeddings_result(region, side, "no_common_subjects")
    k = options["k"]
    if len(idx_a) <= k:
        return _skipped_embeddings_result(region, side, "too_few_subjects")

    paired_a = emb_a[idx_a]
    paired_b = emb_b[idx_b]
    paired_subjects = [ids_a[i] for i in idx_a]

    # ── CKA and kNN ───────────────────────────────────────────────────────
    cka = linear_cka(paired_a, paired_b)
    knn = knn_overlap(paired_a, paired_b, k)

    # ── UMAP ──────────────────────────────────────────────────────────────
    umap_mode = options["umap_a"] is not None
    umap_coords = None
    umap_skipped: list = []
    if umap_mode:
        proj = _compute_umap_projection(options["umap_a"], options["umap_b"], region, side, paired_a, paired_b)
        if isinstance(proj, str):
            umap_skipped.append({"region": region, "side": side, "reason": proj})
        else:
            umap_coords = proj  # (umap_a, umap_b_aligned)

    # ── Covariate ─────────────────────────────────────────────────────────
    covariate_map: "dict | None" = options.get("covariate")
    covariate_mode = covariate_map is not None

    # ── Build rows ────────────────────────────────────────────────────────
    rows: list[dict] = []
    for n, (subject, knn_val) in enumerate(zip(paired_subjects, knn)):
        row: dict = {"region": region, "side": side, "subject": subject, "knn_overlap": repr(float(knn_val))}
        if umap_mode:
            if umap_coords is not None:
                ua, ub = umap_coords
                ax_, ay_ = float(ua[n, 0]), float(ua[n, 1])
                bx_, by_ = float(ub[n, 0]), float(ub[n, 1])
                disp = float(np.hypot(ax_ - bx_, ay_ - by_))
                row["umap_a_x"] = repr(ax_)
                row["umap_a_y"] = repr(ay_)
                row["umap_b_x"] = repr(bx_)
                row["umap_b_y"] = repr(by_)
                row["displacement"] = repr(disp)
            else:
                for col in _EMB_UMAP_COLUMNS:
                    row[col] = ""
        if covariate_mode:
            val = covariate_map.get(subject)
            row[_EMB_COVARIATE_COLUMN] = repr(float(val)) if val is not None else ""
        rows.append(row)

    # ── PNG ───────────────────────────────────────────────────────────────
    if umap_mode and umap_coords is not None:
        png_path = Path(options["output"]) / f"umap_{region}_{side}.png"
        cov_vals = None
        if covariate_mode:
            cov_vals = [covariate_map.get(s) for s in paired_subjects]
        _save_umap_comparison_png(
            png_path, region, side, umap_coords[0], umap_coords[1], cov_vals, options.get("covariate_column")
        )

    # ── Summary ───────────────────────────────────────────────────────────
    summary = _summarise_embeddings_region(
        rows=rows,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        cka=cka,
        n_features_a=paired_a.shape[1],
        n_features_b=paired_b.shape[1],
        umap_mode=umap_mode,
        umap_coords=umap_coords,
        subjects=paired_subjects,
        covariate_map=covariate_map,
    )

    return {
        "key": key,
        "region": region,
        "side": side,
        "rows": rows,
        "summary": summary,
        "skipped": None,
        "umap_skipped": umap_skipped,
    }
