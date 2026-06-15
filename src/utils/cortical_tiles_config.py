import json
import os
from dataclasses import dataclass


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
        with open(path, "r") as f:
            config = json.load(f)
        masks_version = config.get("masks_version", "canonical_25")
        return CorticalTilesConfig(masks_version=masks_version, out_voxel_size=2.0)


def versioned_crops_exist(output_dir: str, masks_version: str, voxel_size: float) -> bool:
    vox_str = f"{int(voxel_size)}mm"
    crops_path = os.path.join(output_dir, "crops", masks_version, vox_str)
    if not os.path.isdir(crops_path):
        return False
    return any(True for _ in os.scandir(crops_path))
