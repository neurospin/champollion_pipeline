import errno

from os import strerror
from os.path import exists

def does_folder_exists_or_except(file: str) -> None:
     if not exists(file):
        raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), file)
     

def are_paths_valid_or_except(paths: list[str]) -> None:
   try: 
      for p in paths:
          does_folder_exists_or_except(p)
   except FileNotFoundError as e:
       raise e