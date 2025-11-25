from argparse import ArgumentParser
from os import getcwd

from utils.lib import are_paths_valid

def run_deep_folding(input_path: str, output_path: str) -> None:
    print(f"run_deep_folding.py/input: {input_path}")
    print(f"run_deep_folding.py/output: {output_path}")

    if not are_paths_valid([input_path, output_path]):
        raise ValueError("run_deep_folding.py: Please input valid paths.")
    
    local_dir: str = getcwd()
    


def main() -> None :
    parser: ArgumentParser = ArgumentParser(
        prog="run_deep_folding",
        description="Generating sulcal regions with deep_folding from Morphologist's graphs."
        )
    
    parser.add_argument("input", help="Absolute path to Morphologit's graphs.")
    parser.add_argument("output", help="Absolute path to the generated sulcal regions from deep_folding.")

    args = parser.parse_args()

    run_deep_folding(args.input, args.output)

if __name__ == "__main__":
    main()