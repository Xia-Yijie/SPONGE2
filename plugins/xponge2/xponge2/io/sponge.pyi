from collections.abc import Sequence
from typing import Any

from .._core import GaffParameters


def save_sponge_input(
    molecule: Any,
    output_dir: str = ".",
    parameters: GaffParameters | None = None,
    prefix: str = "xponge",
    box: Sequence[float] | None = None,
) -> None: ...
