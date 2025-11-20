from os import getcwd
from argparse import ArgumentParser

def run_morpho_graphs(input_path: str) -> None:
    return None

def main() -> None:

    print(f"Current working directory: {getcwd()}")
    parser: ArgumentParser = ArgumentParser(
        prog="morphologist_graphs_generator",
        description="Generating graphs with morphologist from the user raw data."
        )
    
    parser.add_argument("input", help="Path to the user's raw data.")
    parser.add_argument("--output-dir", help="Path to the generated graphs from morphologist.")
    parser.add_argument("--morphologist-path", help="Path to Morphologist.")

    args = parser.parse_args()

    print(f"morphologist_graphs_generator.py/main/args/input: {(args.input)}")


if __name__ == "__main__":
    main()