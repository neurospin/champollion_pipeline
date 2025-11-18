import subprocess, sys

def main() -> None:
    try:
        is_pixi_installed = subprocess.check_output("unknown --version", 
                                                     shell=True, 
                                                     executable="/bin/bash", 
                                                     stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as cpe:
        is_pixi_installed = (cpe.output, False)
    finally:
        if isinstance(is_pixi_installed, tuple):
            print("Error")
        else:
            print(f"return type is: {type(is_pixi_installed)}")
            for l in is_pixi_installed.splitlines():
                print(f"return value is {l.decode()}")

    return None

if __name__ == "__main__":
    main()