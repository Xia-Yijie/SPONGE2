#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PACKAGE = ROOT / "plugins" / "xponge2" / "xponge2"


def _default_compiler() -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    compiler = Path(conda_prefix) / "bin" / "x86_64-conda-linux-gnu-c++"
    if compiler.exists():
        return str(compiler)
    return None


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _find_extension(build_dir: Path) -> Path:
    candidates = sorted(
        build_dir.rglob("_core*.so"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no _core*.so produced under {build_dir}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the SPONGE/xponge native Python extension."
    )
    parser.add_argument(
        "--build-dir",
        default=f"build-xponge-{os.environ.get('PIXI_ENVIRONMENT_NAME', 'local')}",
    )
    parser.add_argument(
        "--parallel",
        default=os.environ.get("XPONGE_PARALLEL", "none"),
        help="SPONGE PARALLEL backend passed to CMake.",
    )
    parser.add_argument(
        "--jobs",
        default=os.environ.get("XPONGE_BUILD_JOBS", "4"),
        help="parallel build jobs",
    )
    args = parser.parse_args()

    build_dir = ROOT / args.build_dir
    configure = [
        "cmake",
        "-S",
        ".",
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DPARALLEL={args.parallel}",
        "-DTARGETS=xponge_core",
        f"-DCMAKE_INSTALL_PREFIX={os.environ.get('CONDA_PREFIX', sys.prefix)}",
    ]
    compiler = _default_compiler()
    if compiler is not None:
        configure.append(f"-DCMAKE_CXX_COMPILER={compiler}")

    _run(configure)
    _run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "xponge_core",
            "--parallel",
            str(args.jobs),
        ]
    )

    extension = _find_extension(build_dir)
    target = PLUGIN_PACKAGE / "_core.abi3.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(extension, target)
    print(f"copied {extension} -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
