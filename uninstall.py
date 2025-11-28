import os
import shutil
import subprocess
from subprocess import run, CalledProcessError
from argparse import ArgumentParser
from os.path import join, exists, expanduser

def remove_pixi_global():
    """Remove global pixi installation and environments."""
    pixi_dir = expanduser("~/.pixi")
    if exists(pixi_dir):
        print(f"Removing global pixi installation at {pixi_dir}...")
        shutil.rmtree(pixi_dir)

    # Remove pixi from bashrc
    bashrc_path = expanduser("~/.bashrc")
    if exists(bashrc_path):
        with open(bashrc_path, "r") as f:
            lines = f.readlines()

        # Remove lines containing "pixi" or ".pixi"
        pixi_lines = [i for i, line in enumerate(lines) if "pixi" in line or ".pixi" in line]
        if pixi_lines:
            print(f"Removing pixi-related lines from {bashrc_path}...")
            with open(bashrc_path, "w") as f:
                for i, line in enumerate(lines):
                    if i not in pixi_lines:
                        f.write(line)

def remove_pixi_env(install_dir):
    """Remove project-specific pixi environment and pixi.toml."""
    pixi_dir = join(install_dir, ".pixi")
    pixi_toml = join(install_dir, "pixi.toml")
    if exists(pixi_dir):
        print(f"Removing project pixi environment at {pixi_dir}...")
        shutil.rmtree(pixi_dir)
    if exists(pixi_toml):
        print(f"Removing pixi.toml at {pixi_toml}...")
        os.remove(pixi_toml)

def remove_repos(install_dir):
    """Remove cloned repositories."""
    repos = ["deep_folding", "champollion_V1"]
    for repo in repos:
        repo_path = join(install_dir, repo)
        if exists(repo_path):
            print(f"Removing {repo} at {repo_path}...")
            shutil.rmtree(repo_path)

def remove_data_dir(install_dir):
    """Remove the data directory."""
    data_dir = join(install_dir, "data")
    if exists(data_dir):
        print(f"Removing data directory at {data_dir}...")
        shutil.rmtree(data_dir)

def main(installation_dir: str, remove_all: bool = False) -> None:
    """Uninstall Champollion V1 pipeline and pixi."""
    abs_install_dir = os.path.abspath(installation_dir)

    # Remove global pixi
    remove_pixi_global()

    # Remove project pixi environment
    remove_pixi_env(abs_install_dir)

    # Remove repositories
    remove_repos(abs_install_dir)

    # Remove data directory
    remove_data_dir(abs_install_dir)

    # Optionally remove the entire installation directory
    if remove_all:
        print(f"Removing installation directory {abs_install_dir}...")
        shutil.rmtree(abs_install_dir)
        print("Uninstallation complete.")
    else:
        print("Uninstallation complete (installation directory preserved).")

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="uninstall.py",
        description="Uninstall the Champollion V1 pipeline and pixi."
    )
    parser.add_argument(
        "--installation_dir",
        help="Absolute path to the installation location",
        type=str,
        default="."
    )
    parser.add_argument(
        "--remove-all",
        help="Remove the entire installation directory",
        action="store_true"
    )
    args = parser.parse_args()

    main(args.installation_dir, args.remove_all)