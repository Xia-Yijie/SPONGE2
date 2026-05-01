from typing import Any

from .._core import GaffParameters

_FrcmodSections = tuple[str, str, str, str, str, str, dict[str, Any]]

def load_gaff_parameters(
    dat_path: str | None = None, frcmod_path: str | None = None
) -> GaffParameters: ...
def load_frcmod(filename: str) -> _FrcmodSections: ...
