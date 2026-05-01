from pathlib import Path

def parmchk2_gaff(
    ifname: str | Path,
    ofname: str | Path,
    *,
    ffset: int = 1,
    print_all: bool = False,
    print_dihedral_contain_X: bool = True,
    datapath: str | Path | None = None,
) -> None: ...
