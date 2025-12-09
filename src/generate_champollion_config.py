from subprocess import run
from argparse import ArgumentParser

from os import getcwd
from os import chdir

from os.path import dirname
from os.path import exists
from os.path import join


def handle_yaml_conf(conf_loc: str, dataset_loc: str):
    """Loads the yaml configuration file and returns it."""

    lines: list[str] = list()
    
    with open(conf_loc, "r") as f:
        for line in f.readlines():
            if "dataset_folder" in line:
                lines.append(f"dataset_folder: {dirname(dataset_loc)}")
            else:
                lines.append(line)

    with open(conf_loc, "w") as f:
        f.writelines(lines)


def main(dataset: str, champollion_dir: str, crops_dir: str) -> None:

    if not exists(crops_dir):
        raise ValueError(f"generate_chamollion_config: Please input correct values. {crops_dir} does not exists.")

    local_dir: str = getcwd()
    dataset_loc: str = join(local_dir, f"{champollion_dir}contrastive/configs/dataset/{dataset}")
    if not exists(dataset_loc):
        run(["mkdir", "-p", dataset_loc], check=True)
    if not exists(join(dataset_loc, "reference.yaml")):
        run(["cp", "../reference.yaml", dataset_loc], check=True)
    
    chdir(dataset_loc)
    
    my_lines: list[str] = list()
    with open("reference.yaml", 'r') as f:
        for line in f.readlines():
            computed_path: str = f"{dataset}/derivatives/deep_folding-2025"
            my_lines.append(line.replace("TESTXX", computed_path))
    
    with open("reference.yaml", "w") as f:
        f.writelines(my_lines)
    
    chdir(champollion_dir)
    print(f"generate_champollion_config.py/main/chdir: {getcwd()}")
    run(["python3", "./contrastive/utils/create_dataset_config_files.py", "--path", dataset_loc, "--crop_path", crops_dir], check=True)

    handle_yaml_conf("./contrastive/configs/dataset_localization/local.yaml", dataset_loc)

    chdir(local_dir)

    return None


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        prog=__file__,
        description="Defining and generating Champollion's configuration."
    )
    parser.add_argument(
        "crop_path",
        help="Absolute path to crops path.",
        type=str
    )
    parser.add_argument(
        "--dataset",
        help="Name of the dataset.",
        type=str,
        required=True)
    parser.add_argument(
        "--champollion_loc",
        help="Absolute path to Champollion binanries.",
        type=str,
        default=join(getcwd(), "../../champollion_V1/")
    )
    args = parser.parse_args()
    
    main(args.dataset, args.champollion_loc, args.crop_path)