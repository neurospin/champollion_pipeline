from subprocess import run
from subprocess import CalledProcessError

from sys import stderr

from os import getcwd
from os import chdir
from os.path import join
from os.path import isabs
from os.path import abspath

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
    
def get_absolute_path(path):
    if isabs(path):
        return path
    else:
        abs_path = abspath(path)
        return abs_path

def main(installation_dir: str) -> None:

    local_dir: str = getcwd()
    abs_install_dir = get_absolute_path(installation_dir)
    chdir(abs_install_dir)

    link_to_deep_folding_repo: str = "https://github.com/neurospin/deep_folding.git"
    link_to_champollion_repo: str = "https://github.com/neurospin/champollion_V1.git"

    # Pixi part
    if not is_pixi_installed():
        print("Installing pixi: ")
        run("curl -fsSL https://pixi.sh/install.sh | bash", shell=True, executable="/bin/bash")
        run("source ~/.bashrc", shell=True, executable="/bin/bash")

    run("pixi init -c https://brainvisa.info/neuro-forge -c pytorch -c nvidia -c conda-forge", shell=True, executable="/bin/bash")
    with open("pixi.toml", mode="a") as conf:
        conf.write('soma-env = ">=0.0"\n')
        conf.write('libjpeg-turbo = {channel= "conda-forge", version= ">=3.0"}\n')
        conf.write('\n')
        conf.write('[pypi-dependencies]\n')
        conf.write('dracopy = ">=1.4.2"\n')
    
    run("pixi add anatomist morphologist soma-env=0.0 pip", shell=True, executable="/bin/bash")

    #Git part
    run(f"git clone {link_to_deep_folding_repo}", shell=True, executable="/bin/bash")
    run(f"git clone {link_to_champollion_repo}", shell=True, executable="/bin/bash")

    #software installation part
    chdir(join(abs_install_dir, 'deep_folding'))
    run("SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip3 install -e .", shell=True, executable="/bin/bash")
    chdir(join(abs_install_dir, "champollion_V1"))
    run("pip3 install -e .", shell=True, executable="/bin/bash")
    
    #Creating the default data file
    run("mkdir data")
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