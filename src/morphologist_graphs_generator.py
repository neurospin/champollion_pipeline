import errno

from os import getcwd
from os import strerror
from os.path import exists

from subprocess import run

from argparse import ArgumentParser

def run_morpho_graphs(input_path: str):
    if not exists(input_path):
        raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), input_path)
    else:
        return input_path

def main() -> None:

    print(f"Current working directory: {getcwd()}")
    parser: ArgumentParser = ArgumentParser(
        prog="morphologist_graphs_generator",
        description="Generating graphs with morphologist from the user raw data."
        )
    
    parser.add_argument("input", help="Path to the user's raw data.")
    parser.add_argument("--output-dir", help="Path to the generated graphs from morphologist. Default is the parent directory")
    parser.add_argument("--morphologist-path", help="Path to Morphologist.")

    args = parser.parse_args()

    print(f"morphologist_graphs_generator.py/main/args/input: {(args.input)}")


if __name__ == "__main__":
    main()