"""Region model directory layout shared by the tests/test_generate_embeddings*.py topic files
(TASK-CODE-BDRABCZUK-73BD87657EA2)."""


def make_region_logs(model_dir) -> None:
    """Create model_dir/logs/ holding an epoch=80 checkpoint, so the region has weights to evaluate."""
    ckpt_dir = model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=80-step=1.ckpt").write_text("placeholder")
