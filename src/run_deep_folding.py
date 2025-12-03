import pprint


from argparse import ArgumentParser

from os import chdir
from os import getcwd
from os.path import dirname
from os.path import join

from joblib import cpu_count

from subprocess import run

from utils.lib import are_paths_valid

def run_deep_folding(input_path: str, output_path: str, path_to_graph, path_sk_with_hull, sk_qc_path, njobs) -> None:
    print(f"run_deep_folding.py/input: {input_path}")
    print(f"run_deep_folding.py/output: {output_path}")

    if not are_paths_valid([input_path, output_path]):
        raise ValueError("run_deep_folding.py: Please input valid paths.")
    
    local_dir: str = getcwd()
        
        # Moving to deep_folding's script location
    chdir(join(dirname(dirname(local_dir)), 'deep_folding/deep_folding/brainvisa/'))

    path_to_graph_arg = (
        f" --path_to_graph {path_to_graph}" 
        if path_to_graph 
        else "--path_to_graph t1mri/default_acquisition/default_analysis/folds/3.1"
        )
    
    path_sk_with_hull_arg = (
        f" --path_sk_with_hull {path_sk_with_hull}" 
        if path_sk_with_hull 
        else "--path_sk_with_hull t1mri/default_acquisition/default_analysis/segmentation"
        )
    sk_qc_path_arg = f"--sk_qc_path {sk_qc_path}" if sk_qc_path else ""

    if njobs >= cpu_count():
        print(f"run_deep_folding.py/run_deep_folding/njobs: Warning you are trying to run more jobs than you have cores, the script will run with {cpu_count() - 2} jobs.")

    run(f"python3 generate_sulcal_regions.py -d {input_path}"
        f" {path_to_graph_arg}" 
        f" {path_sk_with_hull_arg}"
        f" {njobs}", 
        shell=True, 
        executable="/bin/bash")
    
    chdir(local_dir)
    
    


def main() -> None :
    parser: ArgumentParser = ArgumentParser(
        prog="run_deep_folding",
        description="Generating sulcal regions with deep_folding from Morphologist's graphs."
        )
    
    parser.add_argument("input", help="Absolute path to Morphologit's graphs.")
    parser.add_argument("output", help="Absolute path to the generated sulcal regions from deep_folding.")
    parser.add_argument("--region-file", help="Absolute path to the user's sulcal region's configuration file.")
    parser.add_argument(
        "--path_to_graph", type=str, required=True, help="Contains the sub-path that, for each subject, permits getting the sulcal graphs."
    )
    parser.add_argument(
        "--path_sk_with_hull", type=str, required=True, help="Contains the sub-path where to get the skeleton with hull."
    )
    parser.add_argument(
        "--sk_qc_path", type=str, default="", help="the path to the QC file if it exists (the format of the QC file is given below)"
    )
    parser.add_argument(
        "--njobs", help="Number of CPU cores allowed to use. Default is your maximum number of cores - 2 or up to 22 if you have enough cores.",
        type=int,
        default=min(22, cpu_count() - 2)
    )

    args = parser.parse_args()

    run_deep_folding(args.input, args.output, args.path_to_graph, args.path_sk_with_hull, args.sk_qc_path, args.njobs)

if __name__ == "__main__":
    main()