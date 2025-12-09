import pprint
import sys

from argparse import ArgumentParser

from os import chdir
from os import getcwd
from os.path import dirname
from os.path import join

from subprocess import run, check_call

from joblib import cpu_count

from subprocess import run

from utils.lib import are_paths_valid

def put_together_embeddings(embeddings_subpath: str, output_path: str, path_models: str) -> None:
    print(f"put_together_embeddings.py/input: {embeddings_subpath}")
    print(f"put_together_embeddings.py/input: {path_models}")
    print(f"put_together_embeddings.py/output: {output_path}")

    run(["mkdir", "-p", f"{output_path}"])

    if not are_paths_valid([output_path, path_models]):
        raise ValueError("put_together_embeddings.py: Please input valid paths. "
			 f"Given absolute paths are: {output_path}")
    
    local_dir: str = getcwd()
        
    # Moving to deep_folding's script location
    chdir(join(dirname(dirname(local_dir)), 'champollion_V1/contrastive/utils'))

    cmd = [sys.executable,
         f"put_together_embeddings_files.py",
         "--embeddings_subpath", f"{embeddings_subpath}",
         "--output_path", f"{output_path}",
         "--path_models", f"{path_models}",]
    
    try:
        check_call(cmd)
    except Exception as e:
        print(f"Error running command: {e}")
        sys.exit(1)
    
    chdir(local_dir)
    
    


def main() -> None :
    parser: ArgumentParser = ArgumentParser(
        prog="put_together_embeddings",
        description="Put together embeddings of Champollion_V1 in a single folder."
        )
    
    parser.add_argument(
        "--embeddings_subpath", type=str, required=True, help="Sub-path to embeddings inside model folder."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Folder where to put all embeddings"
    )
    parser.add_argument(
        "--path_models",
        type=str,
        default="/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation",
        help="Path where all models lie."
    )

    args = parser.parse_args()

    put_together_embeddings(args.embeddings_subpath, args.output_path, args.path_models)

if __name__ == "__main__":
    main()
