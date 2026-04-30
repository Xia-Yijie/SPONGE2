import filecmp
import importlib
import json
import os
from pathlib import Path

import pytest


def _load_reference_payload():
    statics_path = Path(__file__).resolve().parent.parent / "statics"
    reference_path = statics_path / "pubchem_gaff_1000" / "reference.json"
    if not reference_path.exists():
        pytest.skip(
            "Missing xponge PubChem GAFF reference. "
            "Run `pixi run -e ref-gen ref-gen-xponge`.",
            allow_module_level=True,
        )
    payload = json.loads(reference_path.read_text())
    entries = payload.get("entries", [])
    case_limit = os.environ.get("XPONGE_BENCH_CASE_LIMIT")
    if case_limit:
        entries = entries[: int(case_limit)]
    return reference_path.parent, payload, entries


REFERENCE_ROOT, REFERENCE_PAYLOAD, REFERENCE_ENTRIES = _load_reference_payload()


def _load_xponge_module():
    module_name = os.environ.get("XPONGE_BENCH_MODULE", "xponge2")
    return importlib.import_module(module_name)


def _write_gaff_mol2(xponge_module, input_mol2_path, output_mol2_path):
    assign = xponge_module.get_assignment_from_mol2(str(input_mol2_path))
    if hasattr(assign, "determine_atom_type"):
        assign.determine_atom_type("gaff")
    else:
        assign.Determine_Atom_Type("gaff")

    if hasattr(assign, "save_as_mol2"):
        assign.save_as_mol2(str(output_mol2_path), atomtype="gaff")
    else:
        assign.Save_As_Mol2(str(output_mol2_path), atomtype="gaff")


@pytest.fixture(scope="module")
def xponge_module():
    try:
        return _load_xponge_module()
    except ImportError as exc:
        pytest.skip(f"xponge benchmark module is not available: {exc}")


@pytest.fixture(scope="module")
def pubchem_run_dir(outputs_path):
    run_dir = Path(outputs_path) / "pubchem_gaff_1000"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def test_pubchem_gaff_reference_has_expected_count():
    expected_count = int(
        REFERENCE_PAYLOAD.get("metadata", {}).get("target_count", 1000)
    )
    assert len(REFERENCE_PAYLOAD.get("entries", [])) >= expected_count


@pytest.mark.parametrize(
    "entry",
    REFERENCE_ENTRIES,
    ids=lambda entry: entry["case_name"],
)
def test_pubchem_gaff_mol2_matches_ambertools_reference(
    entry,
    pubchem_run_dir,
    xponge_module,
):
    input_mol2 = REFERENCE_ROOT / entry["input_mol2_file"]
    reference_mol2 = REFERENCE_ROOT / entry["reference_mol2_file"]
    output_mol2 = pubchem_run_dir / f"{entry['case_name']}.mol2"

    assert input_mol2.exists(), f"Missing input mol2 for {entry['case_name']}"
    assert reference_mol2.exists(), (
        f"Missing reference mol2 for {entry['case_name']}"
    )

    _write_gaff_mol2(xponge_module, input_mol2, output_mol2)
    assert filecmp.cmp(output_mol2, reference_mol2, shallow=False), (
        f"{entry['case_name']} mol2 output differs from AmberTools reference"
    )
