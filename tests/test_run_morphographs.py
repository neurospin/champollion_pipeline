import pytest

from os import getcwd
from os.path import exists

from src.generate_morphologist_graphs import run_morpho_graphs

path: str = "/my/path/example/"

def test_does_path_exists ():

    # Testing input path
    try:
        run_morpho_graphs(path)
        assert 0 != 0 # made to fail even if path exists
    except FileNotFoundError as e:
        assert isinstance(e, FileNotFoundError) == True
        assert  "[Errno 2] No such file or directory: '/my/path/example/" in str(e) 

    # Testing output path
    try: 
        run_morpho_graphs(".", path)
        assert 0 != 0
    except FileNotFoundError as e:
        assert isinstance(e, FileNotFoundError) == True
        assert "[Errno 2] No such file or directory: '/my/path/example/" in str(e) 

def test_does_morphologist_create_outputs():

    run_morpho_graphs(path)

    assert exists(f"{getcwd()}/../../derivatives/morphologist-5.2/")

def test_are_output_in_correct_location ():

    # Using local data for testing, need to adapt to your environment
    local_input = f"{getcwd()}/../../runs/TEST_TEMPLATE/rawdata/"
    local_output = f"{getcwd()}/../../runs/TEST08/derivatives/"
    
    #calling with local paths
    run_morpho_graphs(local_input, local_output)

    assert exists(f"{local_output}/morphologist-5.2/")