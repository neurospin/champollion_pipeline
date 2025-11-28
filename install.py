from subprocess import run
from subprocess import CalledProcessError

from sys import stderr

from os import getcwd
from os import chdir
from os import pathsep
from os import environ
from os.path import join
from os.path import isabs
from os.path import abspath
from os.path import exists

from argparse import ArgumentParser

def is_pixi_installed():
    try:
        result = run(
            ["pixi", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except (CalledProcessError, FileNotFoundError):
        return False
    
def get_absolute_path(path: str):
    return abspath(path) if not isabs(path) else path
    
def run_pixi(command: str, env=None):
    env = (env or environ).copy()
    env["PATH"] = f"{join(environ['HOME'], '.pixi', 'bin')}{pathsep}{env['PATH']}"
    run(command.split(), check=True, env=env)

def run_in_pixi(command):
    env = environ.copy()
    env["PATH"] = f"{join(environ['HOME'], '.pixi', 'bin')}{pathsep}{env['PATH']}"
    run(["pixi", "run", *command.split()], check=True, env=env)


def clone_repo(url: str, dirname):
    if not exists(dirname):
        run(["git", "clone", url, dirname], check=True)
    else:
        print(f"Directory {dirname} already exists, skipping clone.")


def main(installation_dir: str) -> None:

    local_dir: str = getcwd()
    abs_install_dir = get_absolute_path(installation_dir)
    chdir(abs_install_dir)

    link_to_deep_folding_repo: str = "https://github.com/neurospin/deep_folding.git"
    link_to_champollion_repo: str = "https://github.com/neurospin/champollion_V1.git"

    # Pixi part
    if not is_pixi_installed():
        print("Installing pixi: ")
        # Download and run the install script
        run(["bash", "-c", "curl -fsSL https://pixi.sh/install.sh | bash"], check=True)
        environ["PATH"] = f"{join(environ['HOME'], '.pixi', 'bin')}{pathsep}{environ['PATH']}"

        # Source ~/.bashrc in the current shell (if needed)
        # Note: This only affects the current subprocess, not the parent shell.
        run(
            ["bash", "-c", "source ~/.bashrc && env"],
            check=True,
            text=True,
        )

    #TO_REMOVE
    print("install.py/main/pixi init")

    run(["bash", "-c",] + ' '.split("pixi init -c https://brainvisa.info/neuro-forge -c pytorch -c nvidia -c conda-forge"), check=True)
    with open("pixi.toml", mode="a") as conf:
        conf.write('soma-env = ">=0.0"\n')
        conf.write('libjpeg-turbo = {channel= "conda-forge", version= ">=3.0"}\n')
        conf.write('\n')
        conf.write('[pypi-dependencies]\n')
        conf.write('dracopy = ">=1.4.2"\n')
    
    #TO_REMOVE
    print("install.py/main/pixi add")

    run(["bash", "-c",] + ' '.split("pixi add anatomist morphologist soma-env=0.0 pip"), check=True)

    #Git part
    print("install.py/main/git clone")
    clone_repo(link_to_deep_folding_repo, "deep_folding")
    clone_repo(link_to_champollion_repo, "champollion_V1")

    #software installation part
    chdir(join(abs_install_dir, 'deep_folding'))
    run_in_pixi("SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip3 install -e .")
    chdir(join(abs_install_dir, "champollion_V1"))
    run_in_pixi("pip3 install -e .")
    
    #Creating the default data file
    chdir(abs_install_dir)
    run(["mkdir", "-p", "data"], check=True)
    chdir(local_dir)

    return None

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        prog=__file__,
        description="Installing the different dependencies of the Champllion V1 pipeline."
        )
    
    parser.add_argument("--installation_dir", help="Absolute path to the wished installation location", type=str, default=".")

    args = parser.parse_args()
    
    main(args.installation_dir)