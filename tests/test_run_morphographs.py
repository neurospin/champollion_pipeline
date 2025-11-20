import pytest

from src.morphologist_graphs_generator import run_morpho_graphs

def does_morphologist_cli_returns ():
    assert run_morpho_graphs("/my/path/example/") == "/my/path/example"