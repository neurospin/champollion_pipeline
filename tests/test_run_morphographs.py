import pytest

from os import getcwd
from os.path import exists

from src.morphologist_graphs_generator import run_morpho_graphs

def test_does_path_exists ():
    path: str = "/my/path/example/"

    try:
        run_morpho_graphs(path)
        assert 0 != 0
    except FileNotFoundError as e:
        assert isinstance(e, FileNotFoundError) == True
        assert  "[Errno 2] No such file or directory: '/my/path/example/" in str(e) 

def test_does_morphologist_create_outputs():
    path: str = "/my/path/example/"

    run_morpho_graphs(path)

    assert exists(f"{getcwd()}/../../runs/derivatives/morphologist-5.2/")