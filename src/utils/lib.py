import errno

from os import strerror
from os.path import exists

def are_paths_valid(paths: list[str]) -> bool | None:
   if len(paths) <= 0:
        raise ValueError("List doesn't contain any path.")
   else:
      index: int = 0
      is_valid = True
      while is_valid and index < len(paths):
         is_valid = exists(paths[index])
         index += 1
      
      return is_valid