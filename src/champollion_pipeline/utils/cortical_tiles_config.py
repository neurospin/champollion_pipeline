import json
import os
from dataclasses import dataclass

CROPS_DIRNAME = "crops"
DEFAULT_VOXEL_SIZE = 2.0
INPUT_TYPES_DEFAULT = ("skeleton", "foldlabel")


@dataclass
class CorticalTilesConfig:
    masks_version: str
    out_voxel_size: float


class CorticalTilesConfigFactory:
    @classmethod
    def from_args(cls, args) -> "CorticalTilesConfig":
        masks_version = args.masks if args.masks else "canonical_25"
        return CorticalTilesConfig(masks_version=masks_version, out_voxel_size=2.0)

    @classmethod
    def from_pipeline_json(cls, path: str) -> "CorticalTilesConfig | None":
        if not os.path.exists(path):
            return None
        with open(path) as f:
            config = json.load(f)
        masks_version = config.get("masks_version", "canonical_25")
        return CorticalTilesConfig(masks_version=masks_version, out_voxel_size=2.0)


def compute_versioned_crops_dir(
    derivatives_dir: str, masks_version: str, voxel_size: float = DEFAULT_VOXEL_SIZE
) -> str:
    """Return <derivatives_dir>/crops/<masks_version>/<int(voxel_size)>mm.

    Pure: no I/O, no existence check, no validation of masks_version. Complexity: O(1).
    """
    return os.path.join(derivatives_dir, CROPS_DIRNAME, masks_version, f"{int(voxel_size)}mm")


def versioned_crops_exist(output_dir: str, masks_version: str, voxel_size: float) -> bool:
    crops_path = compute_versioned_crops_dir(output_dir, masks_version, voxel_size)
    if not os.path.isdir(crops_path):
        return False
    return any(True for _ in os.scandir(crops_path))
