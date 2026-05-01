import importlib
import json
import os
import shutil
from pathlib import Path

import pytest

from benchmarks.comparison.tests.xponge.ref_gen.generate_amber_biopolymer_inputs import (
    CASES,
    _build_molecule,
    _save_sponge_input,
    parse_sponge_mdout_terms,
    write_run0_mdin,
)
from benchmarks.utils import Outputer, Runner


CASE_NAMES = [case["case_name"] for case in CASES]


def _load_xponge_module():
    module_name = os.environ.get("XPONGE_BENCH_MODULE", "xponge2")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"xponge benchmark module is not available: {exc}")
    missing = [
        name
        for name in ("ResidueType",)
        if not hasattr(module, name)
    ]
    if not (
        hasattr(module, "save_sponge_input")
        or hasattr(getattr(module, "Molecule", object), "save_sponge_input")
    ):
        missing.append("save_sponge_input")
    if missing:
        pytest.skip(
            f"{module_name} does not expose AMBER biopolymer API(s): "
            + ", ".join(missing)
        )
    return module_name, module


def _import_forcefield_modules(module_name, forcefield_modules):
    for forcefield_module in forcefield_modules:
        try:
            importlib.import_module(f"{module_name}.{forcefield_module}")
        except ImportError as exc:
            pytest.skip(
                f"{module_name} forcefield module is not available: "
                f"{forcefield_module}: {exc}"
            )


def _load_case_metadata(statics_path, case_name):
    metadata_path = (
        Path(statics_path) / "amber_biopolymer_inputs" / case_name / "case.json"
    )
    if not metadata_path.exists():
        pytest.skip(
            "Missing xponge AMBER biopolymer reference. "
            "Run `python benchmarks/comparison/tests/xponge/ref_gen/"
            "generate_amber_biopolymer_inputs.py`."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def xponge_runtime():
    return _load_xponge_module()


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_amber_biopolymer_reference_files_are_present(statics_path, case_name):
    metadata = _load_case_metadata(statics_path, case_name)
    case_root = Path(statics_path) / "amber_biopolymer_inputs" / case_name
    reference_root = case_root / "reference"

    assert metadata["reference_kind"] == "sponge_run0_energy_terms"
    assert metadata["energy_terms"]
    missing = [
        file_info["reference_name"]
        for file_info in metadata["files"]
        if not (reference_root / file_info["reference_name"]).exists()
    ]
    assert not missing


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_amber_biopolymer_run0_energy_matches_original_xponge(
    statics_path,
    outputs_path,
    mpi_np,
    xponge_runtime,
    case_name,
):
    module_name, xponge_module = xponge_runtime
    metadata = _load_case_metadata(statics_path, case_name)
    _import_forcefield_modules(module_name, metadata["forcefield_modules"])

    case_dir = Outputer.prepare_output_case(
        statics_path=Path(statics_path) / "amber_biopolymer_inputs",
        outputs_path=Path(outputs_path) / "amber_biopolymer_inputs",
        case_name=case_name,
        run_name=module_name,
    )
    output_dir = case_dir / "generated"

    molecule = _build_molecule(xponge_module, metadata["residues"])
    _save_sponge_input(xponge_module, molecule, metadata["prefix"], output_dir)
    for file_info in metadata["files"]:
        shutil.copy2(
            case_dir / "reference" / file_info["reference_name"],
            output_dir / file_info["name"],
        )
    write_run0_mdin(output_dir, metadata["prefix"])
    Runner.run_sponge(output_dir, mpi_np=mpi_np, mdin_name="sponge.mdin", timeout=120)

    actual_terms = parse_sponge_mdout_terms(output_dir / "mdout.txt")
    reference_terms = metadata["energy_terms"]
    absolute_tolerance = 1.0e-4
    rows = []
    failures = []
    for term, reference_value in reference_terms.items():
        actual_value = actual_terms.get(term)
        diff = (
            float("inf")
            if actual_value is None
            else abs(actual_value - reference_value)
        )
        rows.append(
            [
                term,
                f"{reference_value:.8f}",
                "missing" if actual_value is None else f"{actual_value:.8f}",
                f"{diff:.3e}",
                f"{absolute_tolerance:.1e}",
                "PASS" if diff <= absolute_tolerance else "FAIL",
            ]
        )
        if diff > absolute_tolerance:
            failures.append(term)

    Outputer.print_table(
        ["term", "reference", "actual", "|diff|", "tol", "status"],
        rows,
        title=case_name,
    )
    assert not failures
