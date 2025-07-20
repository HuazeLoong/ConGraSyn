import os
from functools import cache
from PrepareData import algos
from pathlib import Path
from ProcessorData.const import *

@cache
def _get_new_fn_groups(filepath: str = None):
    """Used for debug only"""
    # filepath = Path(filepath) if filepath else os.path.dirname(os.path.realpath(__file__)) + "/new_fg_groups.txt"
    with open(NEW_FG_GROUPS_DIR, 'r') as f:
        new_list = f.readlines()
    new_list = {smiles.strip() for smiles in new_list if smiles.strip()}
    new_list_dict = {f"{index + 144}": smiles for index, smiles in
                     enumerate(new_list) if smiles.strip()}
    return new_list_dict