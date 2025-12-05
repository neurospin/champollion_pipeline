from subprocess import run
from argparse import ArgumentParser

from os import getcwd
from os import chdir

from os.path import basename
from os.path import exists
from os.path import join

def main(loc: str) -> None:
    local_dir: str = getcwd()
    real_conf_loc: str = join(loc, 'champollion_config_data/')
    run(["mkdir", "-p", real_conf_loc])
    run(["cp", "../reference.yaml", real_conf_loc])
    chdir(real_conf_loc)
    my_lines: list[str] = list()
    
    with open("reference.yaml", 'r') as f:
        for line in f.readlines():
            my_lines.append(line.replace("TESTXX", basename(loc)))
    
    with open("reference.yaml", "w") as f:
        f.writelines(my_lines)
    
    
    chdir(local_dir)
    return None


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        prog=__file__,
        description="Defining and generating Champollion's configuration."
    )
    parser.add_argument(
        "config_loc",
        help="Absolute path to the wished configuration's location",
        type=str,
        default="../../data/")
    args = parser.parse_args()
    
    main(args.config_loc)