from importlib.resources import files


def bundled_data_dir():
    return files("xponge2.forcefield.amber.parmchk2") / "data"
