from argparse import ArgumentParser

def main() -> None :
    parser: ArgumentParser = ArgumentParser(
        prog="morphologist_graphs_generator",
        description="Generating graphs with morphologist from the user raw data."
        )
    
    parser.add_argument("input", help="Path to the user's raw data.")
    parser.add_argument("--output-dir", help="Path to the generated graphs from morphologist. Default is the parent directory")

    args = parser.parse_args()

if __name__ == "__name__":
    main()