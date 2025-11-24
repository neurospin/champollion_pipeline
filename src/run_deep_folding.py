from argparse import ArgumentParser

def run_deep_folding(input_path: str, output_path: str) -> None:
    print(f"run_deep_folding.py/input: {input_path}")
    print(f"run_deep_folding.py/output: {output_path}")
    
    return None


def main() -> None :
    parser: ArgumentParser = ArgumentParser(
        prog="run_deep_folding",
        description="Generating sulcal regions with deep_folding from Morphologist's graphs."
        )
    
    parser.add_argument("input", help="Absolute path to Morphologit's graphs.")
    parser.add_argument("output", help="Absolute path to the generated sulcal regions from deep_folding.")

    args = parser.parse_args()

if __name__ == "__name__":
    main()