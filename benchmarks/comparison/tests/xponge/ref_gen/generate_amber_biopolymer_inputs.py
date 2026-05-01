#!/usr/bin/env python3

import argparse
import hashlib
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils import Runner

DEFAULT_XPONGE1_ROOT = Path.home() / "research" / "xponge"
STATICS_ROOT = (
    REPO_ROOT / "benchmarks" / "comparison" / "tests" / "xponge" / "statics"
)
SUITE_NAME = "amber_biopolymer_inputs"

_CAPPED_PROTEIN_RESIDUES = [
    "ACE",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "CYX",
    "GLN",
    "GLU",
    "GLY",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "HIS",
    "NME",
]

CASES = [
    {
        "case_name": "protein_ff14sb_residues",
        "description": "ff14SB capped peptide covering standard protein residue templates",
        "forcefield_modules": ["forcefield.amber.ff14sb"],
        "residues": list(_CAPPED_PROTEIN_RESIDUES),
        "prefix": "ff14sb",
    },
    {
        "case_name": "protein_ff19sb_cmap_residues",
        "description": "ff19SB capped peptide covering protein residue templates and CMAP output",
        "forcefield_modules": ["forcefield.amber.ff19sb"],
        "residues": list(_CAPPED_PROTEIN_RESIDUES),
        "prefix": "ff19sb",
    },
    {
        "case_name": "rna_ol3_terminal_residues",
        "description": "OL3 RNA chain covering terminal and internal nucleotide templates",
        "forcefield_modules": ["forcefield.amber.ol3"],
        "residues": ["A5", "A", "U", "C", "G", "C", "U", "A", "U3"],
        "prefix": "ol3",
    },
    {
        "case_name": "dna_bsc1_terminal_residues",
        "description": "bsc1 DNA chain covering terminal and internal nucleotide templates",
        "forcefield_modules": ["forcefield.amber.bsc1"],
        "residues": ["DA5", "DA", "DT", "DC", "DG", "DC", "DT", "DA", "DA3"],
        "prefix": "bsc1",
    },
]


def _import_forcefields(module_root: str, forcefield_modules):
    for module in forcefield_modules:
        importlib.import_module(f"{module_root}.{module}")


def _build_molecule(xponge_module, residue_names):
    residues = [xponge_module.ResidueType.get_type(name) for name in residue_names]
    molecule = residues[0]
    for residue in residues[1:]:
        molecule += residue
    return molecule


def _save_sponge_input(xponge_module, molecule, prefix: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    save = getattr(xponge_module, "save_sponge_input", None)
    if save is not None and "output_dir" in inspect.signature(save).parameters:
        save(molecule, output_dir, prefix=prefix)
        return
    cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        if save is not None:
            save(molecule, prefix)
        else:
            molecule.save_sponge_input(prefix)
    finally:
        os.chdir(cwd)


ENERGY_TERMS = [
    "bond",
    "angle",
    "dihedral",
    "improper_dihedral",
    "cmap",
    "LJ",
    "LJ_short",
    "Coulomb",
    "PME",
    "PME_excluded",
    "nb14_LJ",
    "nb14_EE",
    "potential",
]


def write_run0_mdin(case_dir: Path, prefix: str):
    (case_dir / "sponge.mdin").write_text(
        "\n".join(
            [
                "xponge AMBER biopolymer run0",
                "mode = nve",
                "step_limit = 0",
                "dt = 0",
                "pbc = 0",
                "cutoff = 999.0",
                f"default_in_file_prefix = {prefix}",
                'frc = "frc.dat"',
                "print_zeroth_frame = 1",
                "write_mdout_interval = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_sponge_mdout_terms(mdout_path: Path):
    lines = mdout_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mdout file: {mdout_path}")

    headers = lines[0].split()
    data_line = None
    for line in reversed(lines[1:]):
        stripped = line.strip()
        if stripped and (stripped[0].isdigit() or stripped[0] in "+-"):
            data_line = stripped
            break
    if data_line is None:
        raise ValueError(f"Missing numeric data row in mdout file: {mdout_path}")

    values = data_line.split()
    if len(headers) != len(values):
        raise ValueError(
            f"mdout header/value mismatch in {mdout_path}: "
            f"{len(headers)} vs {len(values)}"
        )
    kv = dict(zip(headers, values))
    return {
        term: float(kv[term])
        for term in ENERGY_TERMS
        if term in kv
    }


def set_nopbc_coordinate_box(coordinate_path: Path):
    lines = coordinate_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid coordinate file: {coordinate_path}")
    lines[-1] = "1000 1000 1000 90 90 90"
    coordinate_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_case(case, module_root: str, output_root: Path, sponge_cmd: str):
    xponge_module = importlib.import_module(module_root)
    _import_forcefields(module_root, case["forcefield_modules"])

    with tempfile.TemporaryDirectory(prefix=f"{case['case_name']}_") as tmp:
        tmp_path = Path(tmp)
        molecule = _build_molecule(xponge_module, case["residues"])
        _save_sponge_input(xponge_module, molecule, case["prefix"], tmp_path)

        generated_files = sorted(
            path for path in tmp_path.iterdir() if path.is_file()
        )
        for generated_file in generated_files:
            if generated_file.name.endswith("_coordinate.txt"):
                set_nopbc_coordinate_box(generated_file)
        write_run0_mdin(tmp_path, case["prefix"])
        Runner.run_sponge(
            tmp_path,
            mdin_name="sponge.mdin",
            timeout=120,
            sponge_cmd=sponge_cmd,
        )
        energy_terms = parse_sponge_mdout_terms(tmp_path / "mdout.txt")

        case_root = output_root / case["case_name"]
        reference_root = case_root / "reference"
        if case_root.exists():
            shutil.rmtree(case_root)
        reference_root.mkdir(parents=True)

        files = []
        for source in generated_files:
            if not source.name.endswith("_coordinate.txt"):
                continue
            target = reference_root / f"reference_{source.name}"
            shutil.copy2(source, target)
            files.append(
                {
                    "name": source.name,
                    "reference_name": target.name,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "bytes": target.stat().st_size,
                }
            )

        metadata = {
            "case_name": case["case_name"],
            "description": case["description"],
            "forcefield_modules": case["forcefield_modules"],
            "residues": case["residues"],
            "prefix": case["prefix"],
            "reference_module": module_root,
            "reference_kind": "sponge_run0_energy_terms",
            "energy_terms": energy_terms,
            "files": files,
        }
        (case_root / "case.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate old-Xponge references for xponge2 AMBER biopolymer benchmarks."
    )
    parser.add_argument(
        "--xponge-root",
        type=Path,
        default=DEFAULT_XPONGE1_ROOT,
        help="Path to the original xponge repository containing the Xponge package.",
    )
    parser.add_argument(
        "--module",
        default="Xponge",
        help="Reference module name to import after adding --xponge-root to sys.path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=STATICS_ROOT / SUITE_NAME,
        help="Static benchmark output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.xponge_root:
        sys.path.insert(0, str(args.xponge_root))
    sponge_cmd = "SPONGE"
    for case in CASES:
        generate_case(case, args.module, args.output_root, sponge_cmd)


if __name__ == "__main__":
    main()
