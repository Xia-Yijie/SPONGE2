import math

import numpy as np
import pytest

from benchmarks.utils import Extractor, Outputer, Runner
from benchmarks.validation.cv.tests.utils import (
    ANGLE_ATOMS,
    DISTANCE_ATOMS,
    PHI_ATOMS,
    PSI_ATOMS,
    RMSD_ATOMS,
    compute_angle,
    compute_dihedral,
    compute_distance,
    compute_kabsch_rmsd,
    compute_tabulated_distance,
    load_box_lengths,
    load_coordinates,
    perturb_coordinates,
    write_bias_cv_file,
    write_rmsd_reference_file,
    write_validation_mdin,
)
from benchmarks.validation.utils import parse_mdout_rows

ATOM_COUNT = 2129
FORCE_ABS_TOL = 2e-3
PRESSURE_ABS_TOL = 8e-2
STRESS_ABS_TOL = 8e-2
ENERGY_ABS_TOL = 1.5e-2
GRAD_EPS = 1e-5
STRESS_KEYS = ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
STEER_WEIGHT = 2.5
RESTRAIN_WEIGHT = 1.75

CV_SPECS = [
    {"name": "distance", "atoms": DISTANCE_ATOMS},
    {"name": "angle", "atoms": ANGLE_ATOMS},
    {"name": "phi", "atoms": PHI_ATOMS},
    {"name": "psi", "atoms": PSI_ATOMS},
    {
        "name": "combo",
        "atoms": tuple(sorted(set(DISTANCE_ATOMS + ANGLE_ATOMS))),
    },
    {"name": "tab_distance_linear", "atoms": DISTANCE_ATOMS},
    {"name": "rmsd_ala", "atoms": RMSD_ATOMS},
]


@pytest.mark.parametrize(
    ("cv_name", "active_atoms"),
    [(spec["name"], spec["atoms"]) for spec in CV_SPECS],
    ids=[spec["name"] for spec in CV_SPECS],
)
def test_cv_steer_and_restrain_bias_match_theory(
    statics_path, outputs_path, mpi_np, cv_name, active_atoms
):
    baseline_dir, reference_coordinates = _prepare_case(
        statics_path,
        outputs_path,
        mpi_np=mpi_np,
        run_name=f"{cv_name}_bias_baseline",
        target_cv=cv_name,
    )
    coordinates = np.asarray(
        load_coordinates(baseline_dir / "sys_flexible_coordinate.txt"),
        dtype=np.float64,
    )
    box = np.asarray(
        load_box_lengths(baseline_dir / "sys_flexible_coordinate.txt"),
        dtype=np.float64,
    )
    baseline_force = Extractor.extract_sponge_forces(baseline_dir, ATOM_COUNT)
    baseline_terms = _load_scalar_terms(baseline_dir, cv_name)
    cv_value = baseline_terms[cv_name]

    steer_dir, _ = _prepare_case(
        statics_path,
        outputs_path,
        mpi_np=mpi_np,
        run_name=f"{cv_name}_bias_steer",
        target_cv=cv_name,
        steer_weight=STEER_WEIGHT,
    )
    steer_force = Extractor.extract_sponge_forces(steer_dir, ATOM_COUNT)
    steer_terms = _load_scalar_terms(steer_dir, cv_name)
    steer_expected_force = _build_bias_force(
        cv_name,
        coordinates,
        reference_coordinates,
        active_atoms,
        prefactor=-STEER_WEIGHT,
    )
    _assert_force_match(steer_force - baseline_force, steer_expected_force)
    _assert_pressure_stress_match(
        case_name=f"{cv_name}_steer",
        coords=coordinates,
        box=box,
        delta_terms=_diff_terms(steer_terms, baseline_terms),
        expected_force=steer_expected_force,
        expected_energy=STEER_WEIGHT * cv_value,
    )

    restrain_reference = _restrain_reference_for(cv_name, cv_value)
    restrain_dir, _ = _prepare_case(
        statics_path,
        outputs_path,
        mpi_np=mpi_np,
        run_name=f"{cv_name}_bias_restrain",
        target_cv=cv_name,
        restrain_weight=RESTRAIN_WEIGHT,
        restrain_reference=restrain_reference,
    )
    restrain_force = Extractor.extract_sponge_forces(restrain_dir, ATOM_COUNT)
    restrain_terms = _load_scalar_terms(restrain_dir, cv_name)
    restrain_prefactor = (
        -2.0 * RESTRAIN_WEIGHT * (cv_value - restrain_reference)
    )
    restrain_expected_force = _build_bias_force(
        cv_name,
        coordinates,
        reference_coordinates,
        active_atoms,
        prefactor=restrain_prefactor,
    )
    _assert_force_match(
        restrain_force - baseline_force, restrain_expected_force
    )
    _assert_pressure_stress_match(
        case_name=f"{cv_name}_restrain",
        coords=coordinates,
        box=box,
        delta_terms=_diff_terms(restrain_terms, baseline_terms),
        expected_force=restrain_expected_force,
        expected_energy=RESTRAIN_WEIGHT
        * (cv_value - restrain_reference)
        * (cv_value - restrain_reference),
    )


def _prepare_case(
    statics_path,
    outputs_path,
    *,
    mpi_np,
    run_name,
    target_cv,
    steer_weight=None,
    restrain_weight=None,
    restrain_reference=None,
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="alanine_dipeptide_phi_psi",
        mpi_np=mpi_np,
        run_name=run_name,
    )
    reference_coordinates = perturb_coordinates(
        load_coordinates(case_dir / "sys_flexible_coordinate.txt"), RMSD_ATOMS
    )
    write_rmsd_reference_file(
        reference_coordinates, case_dir / "rmsd_ref.txt", RMSD_ATOMS
    )
    write_bias_cv_file(
        case_dir,
        target_cv=target_cv,
        steer_weight=steer_weight,
        restrain_weight=restrain_weight,
        restrain_reference=restrain_reference,
    )
    write_validation_mdin(case_dir, print_pressure=True)
    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)
    return case_dir, reference_coordinates


def _load_scalar_terms(case_dir, cv_name):
    row = parse_mdout_rows(
        case_dir / "mdout.txt",
        columns=("potential", "pressure", *STRESS_KEYS, cv_name),
        int_columns=(),
    )[0]
    return {key: float(value) for key, value in row.items()}


def _diff_terms(biased_terms, baseline_terms):
    return {
        key: float(biased_terms[key] - baseline_terms[key])
        for key in biased_terms
        if key in baseline_terms
    }


def _restrain_reference_for(cv_name, cv_value):
    if cv_name in {"phi", "psi"}:
        return cv_value + 0.6
    if cv_name == "rmsd_ala":
        return cv_value + 0.4
    if cv_name == "tab_distance_linear":
        return cv_value - 0.8
    return cv_value + 0.8


def _build_bias_force(
    cv_name,
    coordinates,
    reference_coordinates,
    active_atoms,
    *,
    prefactor,
):
    gradient = _numerical_gradient(
        cv_name, coordinates, reference_coordinates, active_atoms
    )
    return prefactor * gradient


def _numerical_gradient(
    cv_name, coordinates, reference_coordinates, active_atoms
):
    gradient = np.zeros_like(coordinates, dtype=np.float64)
    for atom_index in active_atoms:
        for axis in range(3):
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[atom_index, axis] += GRAD_EPS
            minus[atom_index, axis] -= GRAD_EPS
            gradient[atom_index, axis] = (
                _evaluate_cv_value(cv_name, plus, reference_coordinates)
                - _evaluate_cv_value(cv_name, minus, reference_coordinates)
            ) / (2.0 * GRAD_EPS)
    return gradient


def _evaluate_cv_value(cv_name, coordinates, reference_coordinates):
    if cv_name == "distance":
        return compute_distance(coordinates, DISTANCE_ATOMS)
    if cv_name == "angle":
        return compute_angle(coordinates, ANGLE_ATOMS)
    if cv_name == "phi":
        return compute_dihedral(coordinates, PHI_ATOMS)
    if cv_name == "psi":
        return compute_dihedral(coordinates, PSI_ATOMS)
    if cv_name == "combo":
        return compute_distance(
            coordinates, DISTANCE_ATOMS
        ) + 0.5 * compute_angle(coordinates, ANGLE_ATOMS)
    if cv_name == "tab_distance_linear":
        return compute_tabulated_distance(
            compute_distance(coordinates, DISTANCE_ATOMS)
        )
    if cv_name == "rmsd_ala":
        return compute_kabsch_rmsd(
            [reference_coordinates[index] for index in RMSD_ATOMS],
            [coordinates[index] for index in RMSD_ATOMS],
        )
    raise ValueError(f"Unsupported CV for bias validation: {cv_name}")


def _assert_force_match(actual_force, expected_force):
    max_abs_err = float(np.max(np.abs(actual_force - expected_force)))
    assert max_abs_err <= FORCE_ABS_TOL


def _assert_pressure_stress_match(
    *,
    case_name,
    coords,
    box,
    delta_terms,
    expected_force,
    expected_energy,
):
    expected_pressure, expected_stress = (
        compute_pressure_stress_from_coords_forces(coords, expected_force, box)
    )
    pressure_abs_err = abs(delta_terms["pressure"] - expected_pressure)
    energy_abs_err = abs(delta_terms["potential"] - expected_energy)
    max_stress_abs_err = max(
        abs(delta_terms[key] - expected_stress[key]) for key in STRESS_KEYS
    )

    Outputer.print_table(
        ["Metric", "Expected", "Actual", "AbsErr", "AbsTol", "Status"],
        [
            [
                "potential",
                f"{expected_energy:.6f}",
                f"{delta_terms['potential']:.6f}",
                f"{energy_abs_err:.6f}",
                f"{ENERGY_ABS_TOL:.6f}",
                "PASS" if energy_abs_err <= ENERGY_ABS_TOL else "FAIL",
            ],
            [
                "pressure",
                f"{expected_pressure:.6f}",
                f"{delta_terms['pressure']:.6f}",
                f"{pressure_abs_err:.6f}",
                f"{PRESSURE_ABS_TOL:.6f}",
                "PASS" if pressure_abs_err <= PRESSURE_ABS_TOL else "FAIL",
            ],
            [
                "max_stress_abs_err",
                "0.000000",
                f"{max_stress_abs_err:.6f}",
                f"{max_stress_abs_err:.6f}",
                f"{STRESS_ABS_TOL:.6f}",
                "PASS" if max_stress_abs_err <= STRESS_ABS_TOL else "FAIL",
            ],
        ],
        title=f"CV Bias Validation: {case_name}",
    )

    assert energy_abs_err <= ENERGY_ABS_TOL
    assert pressure_abs_err <= PRESSURE_ABS_TOL
    assert max_stress_abs_err <= STRESS_ABS_TOL


def compute_pressure_stress_from_coords_forces(coords, forces, box):
    coords = np.asarray(coords, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    volume = float(box[0] * box[1] * box[2])
    scale = 6.946827162543585e4 / volume

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    fx = forces[:, 0]
    fy = forces[:, 1]
    fz = forces[:, 2]

    stress = {
        "Pxx": float(np.sum(x * fx) * scale),
        "Pyy": float(np.sum(y * fy) * scale),
        "Pzz": float(np.sum(z * fz) * scale),
        "Pxy": float(0.5 * np.sum(x * fy + y * fx) * scale),
        "Pxz": float(0.5 * np.sum(x * fz + z * fx) * scale),
        "Pyz": float(0.5 * np.sum(y * fz + z * fy) * scale),
    }
    pressure = (stress["Pxx"] + stress["Pyy"] + stress["Pzz"]) / 3.0
    return pressure, stress
