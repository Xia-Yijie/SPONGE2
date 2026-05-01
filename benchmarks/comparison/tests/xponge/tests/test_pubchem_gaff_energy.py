import json
import shutil
from pathlib import Path

import pytest

from benchmarks.utils import Outputer, Runner


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
    entries = [
        entry for entry in payload.get("entries", []) if "amber_energy" in entry
    ]
    if not entries:
        pytest.skip(
            "Missing Amber energy terms in xponge PubChem GAFF reference. "
            "Run `pixi run -e ref-gen ref-gen-xponge -- --refresh-energy`.",
            allow_module_level=True,
        )
    return reference_path.parent, entries


REFERENCE_ROOT, REFERENCE_ENTRIES = _load_reference_payload()


def _assignment_from_input_mol2(xponge_module, input_mol2_path):
    assign = xponge_module.get_assignment_from_mol2(str(input_mol2_path))
    if hasattr(assign, "determine_atom_type"):
        assign.determine_atom_type("gaff")
    else:
        assign.Determine_Atom_Type("gaff")
    return assign


def _extract_sponge_potential(case_dir):
    lines = (case_dir / "mdout.txt").read_text().splitlines()
    header = lines[0].split()
    values = lines[1].split()
    return float(values[header.index("eff_pot")])


@pytest.fixture(scope="module")
def pubchem_energy_run_dir(outputs_path):
    run_dir = Path(outputs_path) / "pubchem_gaff_energy"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.mark.parametrize(
    "entry",
    REFERENCE_ENTRIES,
    ids=lambda entry: entry["case_name"],
)
def test_pubchem_gaff_xponge_model_energy_matches_amber_reference(
    entry,
    pubchem_energy_run_dir,
    xponge_module,
):
    case_dir = pubchem_energy_run_dir / entry["case_name"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    input_mol2 = REFERENCE_ROOT / entry["input_mol2_file"]
    assign = _assignment_from_input_mol2(xponge_module, input_mol2)
    typed_mol2 = case_dir / "xponge_gaff.mol2"
    frcmod_path = case_dir / "xponge_gaff.frcmod"
    assign.save_as_mol2(str(typed_mol2), atomtype="gaff")
    xponge_module.generate_gaff_frcmod(typed_mol2, frcmod_path)
    residue_type = xponge_module.assignment_to_residue_type(assign, name="MOL")
    molecule = residue_type.to_molecule()
    parameters = xponge_module.load_gaff_parameters(frcmod_path=frcmod_path)
    xponge_module.save_sponge_input(molecule, case_dir, parameters)
    Runner.run_sponge(case_dir, mdin_name="sponge.mdin", timeout=120.0)

    amber_total = float(entry["amber_energy"]["total"])
    sponge_total = _extract_sponge_potential(case_dir)
    abs_diff = abs(sponge_total - amber_total)
    energy_tol = 2.0e-2
    passed = abs_diff <= energy_tol

    Outputer.print_table(
        ["Case", "Amber", "SPONGE", "|dE|", "Status"],
        [
            [
                entry["case_name"],
                f"{amber_total:.6f}",
                f"{sponge_total:.6f}",
                f"{abs_diff:.6e}",
                "PASS" if passed else "FAIL",
            ]
        ],
        title="Xponge GAFF Native Model Energy",
    )

    assert abs_diff <= energy_tol
