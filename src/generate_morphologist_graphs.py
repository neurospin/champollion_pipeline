import errno

from os import getcwd
from os import strerror
from os import listdir
from os import chdir
from os.path import exists
from os.path import isfile
from os.path import join
from os.path import splitext
from os.path import basename

from subprocess import run

from argparse import ArgumentParser

from utils.lib import are_paths_valid
def run_morpho_graphs(input_path: str, output_path: str):
    
    if not are_paths_valid([input_path, output_path]):
        raise ValueError("generate_morphologist_graphs.py: Please input valid paths.")
    
    # List of allowed extensions for files as raw data for the pipeline
    LIST_OF_EXTENSIONS: list[str] = [".nii.gz", ".nii", ".gz"]

    input_files: list[str] = [
        f for f in listdir(input_path) 
                                if isfile(join(input_path, f)) 
                                and splitext(basename(f))[1] in LIST_OF_EXTENSIONS
                                ]

    local_dir: str = getcwd()
    chdir(input_path)
    run(f"morphologist-cli {' '.join(input_files)} {output_path} -- --of morphologist-auto-nonoverlap-1.0", shell=True, executable="/bin/bash")
    chdir(local_dir)

def main() -> None:

    print(f"Current working directory: {getcwd()}")
    parser: ArgumentParser = ArgumentParser(
        prog="morphologist_graphs_generator",
        description="Generating graphs with morphologist from the user raw data."
        )
    
    parser.add_argument("input", help="Absolute path to the user's raw data.")
    parser.add_argument("output", help="Absolute path to the generated graphs from morphologist.\n" \
    "Morphologist will create a $output/derivatives/motphologist-5.2/ directory for output generations.")

    args = parser.parse_args()

    run_morpho_graphs(args.input, args.output)


if __name__ == "__main__":
    main()