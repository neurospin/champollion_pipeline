#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload trained Champollion models to a HuggingFace repository.

Models are stored under {masks_version}/ in the repo so that different
mask set versions coexist in the same repository:

    neurospin/champollion/
        canonical_25/SC-sylv_left/best_model.pt
        canonical_25/SC-sylv_right/best_model.pt
        ...
        canonical_26_1/SC-sylv_left/best_model.pt
        ...

Usage
-----
    python src/upload_models_to_hf.py \\
        /path/to/models_dir \\
        neurospin/champollion \\
        --masks-version canonical_25

    # With an explicit token:
    python src/upload_models_to_hf.py \\
        /path/to/models_dir neurospin/champollion \\
        --masks-version canonical_25 --token hf_xxx
"""

import sys
from os.path import abspath, isdir

from champollion_utils.script_builder import ScriptBuilder


class UploadModelsToHF(ScriptBuilder):
    """Upload a local models directory to a versioned HuggingFace subfolder."""

    def __init__(self):
        super().__init__(
            script_name="upload_models_to_hf",
            description="Upload trained Champollion models to a HuggingFace repository.",
        )
        (self.add_argument(
            "models_dir",
            help="Local directory containing one subfolder per sulcal region "
                 "(e.g. SC-sylv_left/, CINGULATE_right/, …).")
         .add_argument(
            "repo_id",
            help="HuggingFace repository ID (e.g. 'neurospin/champollion').")
         .add_required_argument(
            "--masks-version",
            "Mask version tag used as the upload subfolder in the HF repo "
            "(e.g. 'canonical_25'). Models land at {repo_id}/{masks_version}/.")
         .add_optional_argument(
            "--token",
            "HuggingFace API token. Falls back to the HF_TOKEN env var "
            "or cached login credentials (run 'huggingface-cli login' once).",
            default=None)
         .add_flag(
            "--private",
            "Create the repository as private if it does not yet exist."))

    def run(self) -> int:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("ERROR: huggingface_hub is required. "
                  "Install with: pip install huggingface_hub")
            return 1

        models_dir = abspath(self.args.models_dir)
        if not isdir(models_dir):
            print(f"ERROR: models_dir not found: {models_dir}")
            return 1

        api = HfApi(token=self.args.token)
        masks_version = self.args.masks_version
        repo_id = self.args.repo_id

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=self.args.private,
            exist_ok=True,
        )

        print(f"Uploading  {models_dir}")
        print(f"        →  {repo_id}/{masks_version}/")

        api.upload_folder(
            repo_id=repo_id,
            folder_path=models_dir,
            path_in_repo=masks_version,
            repo_type="model",
        )

        print(f"\nDone. Browse at:")
        print(f"  https://huggingface.co/{repo_id}/tree/main/{masks_version}")
        return 0


def main():
    script = UploadModelsToHF()
    return script.build().print_args().run()


if __name__ == "__main__":
    sys.exit(main())
