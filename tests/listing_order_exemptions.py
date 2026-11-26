"""Exemptions from the listing-order rule of the pipeline (TASK-READORDER-BDRABCZUK-EFCBDE20C624).

tests/test_listing_order_rule.py fails on a glob, iglob, rglob, listdir, scandir, walk or iterdir
call of main.py, src/ or scripts/ that is not inside a sorted() call, unless the call is listed
here with the reason its order never reaches a result (a set, a membership or emptiness test, a
count, ...). It also fails on an entry with an empty reason and on an entry that names no call the
rule reports (a stale entry, left behind after the call was sorted, moved or removed).

Key: (file path relative to the pipeline root, qualified name of the enclosing function or class
or "<module>", ast.unparse() text of the call). No line number: the key stays valid when lines
move. The rule test prints the keys of the calls it reports in this form. Value: the reason, a
non-empty text.
"""

EXEMPTIONS: dict[tuple[str, str, str], str] = {
    (
        "scripts/install_health.py",
        "check_package",
        "src_path.iterdir()",
    ): "read through any(): a presence check (LISTING_PRESENCE_CHECKS)",
    (
        "scripts/install_health.py",
        "check_submodule",
        "abs_path.iterdir()",
    ): "read through any(): a presence check (LISTING_PRESENCE_CHECKS)",
    (
        "src/champollion_pipeline/generate_embeddings.py",
        "GenerateEmbeddings._patch_config_paths",
        "Path(config_path).rglob('*.yaml')",
    ): "each YAML file is read, rewritten and written on its own; only the count of patched files is logged",
    (
        "src/champollion_pipeline/generate_labelling_qc.py",
        "fetch_subject_row",
        "os.listdir(session_dir)",
    ): "only len() of the entries is read (emptiness of the session folder)",
    (
        "src/champollion_pipeline/generate_masks.py",
        "_graph_matches",
        "glob.glob(join(glob.escape(sub['dir']), sub['graph_file']))",
    ): "the filtered candidates are returned through sorted() on the next line",
    (
        "src/champollion_pipeline/models/checkpoint.py",
        "find_epoch_checkpoint",
        "lightning_logs.glob('version_*/checkpoints/*.ckpt')",
    ): "max() over the rank (step, version index, path text), a total order",
    (
        "src/champollion_pipeline/prune_failed_subjects.py",
        "PruneFailedSubjects._discover_subjects",
        "Path(crops_dir).iterdir()",
    ): "set union of the crop ids of each region",
    ("src/champollion_pipeline/prune_failed_subjects.py", "_crop_ids", "side_dir.iterdir()"): "fills a set",
    (
        "src/champollion_pipeline/run_cortical_tiles_inputs.py",
        "has_hemisphere_graph",
        "glob.glob(pattern)",
    ): "read through any(): a presence check",
    (
        "src/champollion_pipeline/run_cortical_tiles_inputs.py",
        "resolve_graph_paths",
        "glob.glob(join(glob.escape(subject_dir), pattern))",
    ): "fills a set; the folders are returned as sorted(rels)",
    (
        "src/champollion_pipeline/snapshot_types/sulcal.py",
        "find_sulcal_graphs",
        "glob.glob(osp.join(glob.escape(root), '**', '*.arg'), recursive=True)",
    ): "filtered by name and acquisition, then graphs.sort() before the return",
    (
        "src/champollion_pipeline/snapshot_types/sulcal.py",
        "list_acquisitions",
        "glob.glob(osp.join(acq_dir, '**', '*.arg'), recursive=True)",
    ): "read through any(): a presence check",
    (
        "src/champollion_pipeline/snapshot_types/sulcal.py",
        "list_acquisitions",
        "os.listdir(t1mri_dir)",
    ): "the kept names are returned as sorted(candidates)",
    (
        "src/champollion_pipeline/utils/archive.py",
        "reuse_cached_extraction",
        "os.scandir(cached_path)",
    ): "read through any(): a presence check (LISTING_PRESENCE_CHECKS)",
    (
        "src/champollion_pipeline/utils/cortical_tiles_config.py",
        "versioned_crops_exist",
        "os.scandir(crops_path)",
    ): "read through any(): a presence check (LISTING_PRESENCE_CHECKS)",
    (
        "src/champollion_pipeline/utils/input_kind.py",
        "_read_folder",
        "os.scandir(folder)",
    ): "the entries are returned sorted by _walk_rank at the end of the function",
    (
        "src/champollion_pipeline/utils/transform_qc.py",
        "transform_qc_rows",
        "os.scandir(side_dir)",
    ): "sorted by entry name on the next line, inside the with block",
    (
        "src/compare_sulcal/crops.py",
        "_list_crop_regions",
        "set_a.iterdir()",
    ): "fills a set; the intersection is returned sorted",
    (
        "src/compare_sulcal/crops.py",
        "_list_crop_regions",
        "set_b.iterdir()",
    ): "fills a set; the intersection is returned sorted",
    (
        "src/compare_sulcal/embeddings.py",
        "_list_embedding_pairs",
        "set_a.glob(f'*{suffix}')",
    ): "fills a set; the intersection is iterated sorted",
    (
        "src/compare_sulcal/embeddings.py",
        "_list_embedding_pairs",
        "set_b.glob(f'*{suffix}')",
    ): "fills a set; the intersection is iterated sorted",
    (
        "src/run_cka.py",
        "RunCKA.run",
        "os.listdir(path_a)",
    ): "regions = sorted(subdirs_a) before any region is used; otherwise only truthiness is read",
}
