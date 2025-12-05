from subprocess import run
from argparse import ArgumentParser

from os import getcwd
from os import chdir

from os.path import basename
from os.path import exists
from os.path import join


def handle_yaml_conf(conf_loc: str, dataset_loc: str):
    """Loads the yaml configuration file and returns it."""

    lines: list[str] = list()
    
    with open(conf_loc, "r") as f:
        for line in f.read():
            lines.append(line)

    for line in lines:
        if "dataset_folder" in line:
            line = f"dataset_folder: {dataset_loc}"

    with open(conf_loc, "w") as f:
        f.writelines(lines)


def main(loc: str, champollion_dir: str, crops_dir: str) -> None:
    local_dir: str = getcwd()
    real_conf_loc: str = join(loc, 'champollion_config_data/')
    if not exists(real_conf_loc):
        run(["mkdir", "-p", real_conf_loc], check=True)
    if not exists(join(real_conf_loc, "reference.yaml")):
        run(["cp", "../reference.yaml", real_conf_loc], check=True)
    
    chdir(real_conf_loc)
    
    my_lines: list[str] = list()
    with open("reference.yaml", 'r') as f:
        for line in f.readlines():
            my_lines.append(line.replace("TESTXX", basename(loc)))
    
    with open("reference.yaml", "w") as f:
        f.writelines(my_lines)
    
    chdir(champollion_dir)
    run(["python3", "./contrastive/utils/create_dataset_config_files.py", "--path", real_conf_loc, "--crop_path", crops_dir], check=True)

    handle_yaml_conf(real_conf_loc, loc)

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
        "--config_loc",
        help="Absolute path to the wished configuration's location",
        type=str,
        default="../../data/")
    parser.add_argument(
        "--champollion_loc",
        help="Absolute path to Champollion binanries.",
        type=str,
        default=join(getcwd(), "../../champollion_V1/")
    )
    args = parser.parse_args()
    
    main(args.config_loc, args.champollion_loc, args.crop_path)