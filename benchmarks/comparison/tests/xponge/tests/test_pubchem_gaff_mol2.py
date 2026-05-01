import filecmp
import json
import os
from pathlib import Path

import pytest

from benchmarks.comparison.tests.xponge.ref_gen.generate_pubchem_gaff import (
    parse_sander_energy_terms,
)


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
def pubchem_run_dir(outputs_path):
    run_dir = Path(outputs_path) / "pubchem_gaff_1000"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def test_pubchem_gaff_reference_has_expected_count():
    expected_count = int(
        REFERENCE_PAYLOAD.get("metadata", {}).get("target_count", 1000)
    )
    assert len(REFERENCE_PAYLOAD.get("entries", [])) >= expected_count


def test_sander_energy_terms_are_parsed(tmp_path):
    sander_out = tmp_path / "sander.out"
    sander_out.write_text(
        """
 NSTEP =        0   TIME(PS) =       0.000  TEMP(K) =     0.00  PRESS =     0.0
 Etot   =      -1.0000  EKtot   =       0.0000  EPtot      =      -1.0000
 BOND   =       1.2500  ANGLE   =       2.5000  DIHED      =       3.7500
 VDWAALS=       4.1250  EELEC   =      -5.2500  EHBOND     =       0.0000
 1-4 NB =       0.6250  1-4 EEL =      -0.8750  RESTRAINT  =       0.0000
"""
    )

    assert parse_sander_energy_terms(sander_out) == {
        "bond": 1.25,
        "angle": 2.5,
        "dihedral": 3.75,
        "vdw": 4.125,
        "elec": -5.25,
        "one_four_vdw": 0.625,
        "one_four_elec": -0.875,
        "total": -1.0,
    }


def test_load_frcmod_api(xponge_module, tmp_path):
    if not hasattr(xponge_module, "load_frcmod"):
        pytest.skip("xponge module does not expose load_frcmod")

    frcmod = tmp_path / "mini.frcmod"
    frcmod.write_text(
        "\n".join(
            [
                "Remark line goes here",
                "MASS",
                "c3 12.010 0.878",
                "",
                "BOND",
                "c3-hc  340.0  1.090",
                "",
                "ANGLE",
                "hc-c3-hc   33.0  109.5",
                "",
                "DIHE",
                "hc-c3-c3-hc   3  0.155  0.0  3.0",
                "",
                "IMPROPER",
                "X -X -c -o    10.5  180.0  2.0",
                "",
                "NONBON",
                "c3  1.9080  0.1094",
                "",
            ]
        )
    )

    atoms, bonds, angles, propers, impropers, ljs, cmap = xponge_module.load_frcmod(frcmod)

    assert "c3\t12.010\tc3" in atoms
    assert "c3-hc\t340.0\t1.090" in bonds
    assert "hc-c3-hc\t33.0\t109.5" in angles
    assert "hc-c3-c3-hc\t0.051666666666666666\t0.0\t3\t1" in propers
    assert "X-X-c-o\t10.5\t180.0\t2" in impropers
    assert "c3-c3\t1.9080\t0.1094" in ljs
    assert cmap == {}


def test_xponge2_config_uses_bundled_gaff_data(xponge_module):
    if not hasattr(xponge_module, "get_config"):
        pytest.skip("xponge module does not expose xponge2 config")

    data_dir = Path(xponge_module.get_config("parmchk2_data_dir"))
    gaff_dat = Path(xponge_module.get_config("gaff_dat_path"))
    gaff2_dat = Path(xponge_module.get_config("gaff2_dat_path"))

    assert data_dir.exists()
    assert gaff_dat == data_dir / "gaff.dat"
    assert gaff2_dat == data_dir / "gaff2.dat"
    assert gaff_dat.exists()
    assert gaff2_dat.exists()


def test_assignment_to_residue_type_api(xponge_module):
    entry = REFERENCE_ENTRIES[0]
    input_mol2 = REFERENCE_ROOT / entry["input_mol2_file"]
    assign = xponge_module.get_assignment_from_mol2(str(input_mol2))
    assign.determine_atom_type("gaff")

    residue_type = xponge_module.assignment_to_residue_type(assign, name="MOL")
    molecule = residue_type.to_molecule()

    assert residue_type.name == "MOL"
    assert len(residue_type.atoms) == int(assign.atom_numbers)
    assert len(molecule.residues) == 1
    assert len(molecule.atoms) == int(assign.atom_numbers)
    bonds_by_index = assign.bonds
    expected_bonds = sum(
        1
        for i in range(int(assign.atom_numbers))
        for j in dict(bonds_by_index[i])
        if i < j
    )
    assert len(molecule.bonds) == expected_bonds


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
