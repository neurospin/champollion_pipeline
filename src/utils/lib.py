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
   

def get_nth_parent_dir(folder: str, n: int) -> str:
    """
    Returns the absolute path to the nth parent of a directory
    
    :param folder: absolute path to a directory
    :type folder: str
    :param n: number of iteration to do
    :type n: int
    :return: absolute path to the nth parent
    :rtype: str
    """

    # Substracting 3 instead of 1 because of the first and last element being ''
    if n >= len(folder.split('/')) - 3:
        return '/' + folder.split('/')[1] + '/'
    else :
        parent_folder: str = folder
        for i in range(n) :
            parent_folder = os.path.dirname(parent_folder)

    return parent_folder