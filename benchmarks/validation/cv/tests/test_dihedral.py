from benchmarks.utils import Outputer, Runner
from benchmarks.validation.utils import parse_mdout_column

from benchmarks.validation.cv.tests.utils import (
    ANGLE_ATOMS,
    DISTANCE_ATOMS,
    PHI_ATOMS,
    PSI_ATOMS,
    RMSD_ATOMS,
    compute_angle,
    compute_distance,
    compute_dihedral,
    compute_kabsch_rmsd,
    compute_tabulated_distance,
    load_coordinates,
    perturb_coordinates,
    write_rmsd_reference_file,
    write_extended_cv_file,
    write_validation_mdin,
)

CASE_NAME = "alanine_dipeptide_phi_psi"
RUN_NAME = "alanine_dipeptide_phi_psi"
DISTANCE_ABS_TOL = 5e-4
ANGLE_ABS_TOL = 5e-4
DIHEDRAL_ABS_TOL = 5e-4
COMBINATION_ABS_TOL = 5e-4
TABULATED_ABS_TOL = 5e-4
RMSD_ABS_TOL = 5e-4


def test_simple_and_composite_cv_matches_geometry(
    statics_path, outputs_path, mpi_np
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=CASE_NAME,
        mpi_np=mpi_np,
        run_name=RUN_NAME,
    )

    reference_coordinates = perturb_coordinates(
        load_coordinates(case_dir / "sys_flexible_coordinate.txt"),
        RMSD_ATOMS,
    )
    write_rmsd_reference_file(
        reference_coordinates, case_dir / "rmsd_ref.txt", RMSD_ATOMS
    )
    write_extended_cv_file(case_dir)
    write_validation_mdin(case_dir)
    Runner.run_sponge(case_dir, timeout=1200, mpi_np=mpi_np)

    coordinates = load_coordinates(case_dir / "sys_flexible_coordinate.txt")
    distance_expected = compute_distance(coordinates, DISTANCE_ATOMS)
    angle_expected = compute_angle(coordinates, ANGLE_ATOMS)
    phi_expected = compute_dihedral(coordinates, PHI_ATOMS)
    psi_expected = compute_dihedral(coordinates, PSI_ATOMS)
    combo_expected = distance_expected + 0.5 * angle_expected
    tabulated_expected = compute_tabulated_distance(distance_expected)
    rmsd_expected = compute_kabsch_rmsd(
        [reference_coordinates[index] for index in RMSD_ATOMS],
        [coordinates[index] for index in RMSD_ATOMS],
    )

    distance_actual = float(
        parse_mdout_column(case_dir / "mdout.txt", "distance")[0]
    )
    angle_actual = float(parse_mdout_column(case_dir / "mdout.txt", "angle")[0])
    phi_series = parse_mdout_column(case_dir / "mdout.txt", "phi")
    psi_series = parse_mdout_column(case_dir / "mdout.txt", "psi")
    phi_actual = float(phi_series[0])
    psi_actual = float(psi_series[0])
    combo_actual = float(parse_mdout_column(case_dir / "mdout.txt", "combo")[0])
    tabulated_actual = float(
        parse_mdout_column(case_dir / "mdout.txt", "tab_distance_linear")[0]
    )
    rmsd_actual = float(
        parse_mdout_column(case_dir / "mdout.txt", "rmsd_ala")[0]
    )

    distance_abs_err = abs(distance_actual - distance_expected)
    angle_abs_err = abs(angle_actual - angle_expected)
    phi_abs_err = abs(phi_actual - phi_expected)
    psi_abs_err = abs(psi_actual - psi_expected)
    combo_abs_err = abs(combo_actual - combo_expected)
    tabulated_abs_err = abs(tabulated_actual - tabulated_expected)
    rmsd_abs_err = abs(rmsd_actual - rmsd_expected)

    Outputer.print_table(
        [
            "Metric",
            "Expected",
            "Actual",
            "AbsErr",
            "AbsTol",
            "Status",
        ],
        [
            [
                "distance",
                f"{distance_expected:.6f}",
                f"{distance_actual:.6f}",
                f"{distance_abs_err:.6f}",
                f"{DISTANCE_ABS_TOL:.6f}",
                "PASS" if distance_abs_err <= DISTANCE_ABS_TOL else "FAIL",
            ],
            [
                "angle",
                f"{angle_expected:.6f}",
                f"{angle_actual:.6f}",
                f"{angle_abs_err:.6f}",
                f"{ANGLE_ABS_TOL:.6f}",
                "PASS" if angle_abs_err <= ANGLE_ABS_TOL else "FAIL",
            ],
            [
                "phi",
                f"{phi_expected:.6f}",
                f"{phi_actual:.6f}",
                f"{phi_abs_err:.6f}",
                f"{DIHEDRAL_ABS_TOL:.6f}",
                "PASS" if phi_abs_err <= DIHEDRAL_ABS_TOL else "FAIL",
            ],
            [
                "psi",
                f"{psi_expected:.6f}",
                f"{psi_actual:.6f}",
                f"{psi_abs_err:.6f}",
                f"{DIHEDRAL_ABS_TOL:.6f}",
                "PASS" if psi_abs_err <= DIHEDRAL_ABS_TOL else "FAIL",
            ],
            [
                "combo",
                f"{combo_expected:.6f}",
                f"{combo_actual:.6f}",
                f"{combo_abs_err:.6f}",
                f"{COMBINATION_ABS_TOL:.6f}",
                "PASS" if combo_abs_err <= COMBINATION_ABS_TOL else "FAIL",
            ],
            [
                "tab_distance_linear",
                f"{tabulated_expected:.6f}",
                f"{tabulated_actual:.6f}",
                f"{tabulated_abs_err:.6f}",
                f"{TABULATED_ABS_TOL:.6f}",
                "PASS" if tabulated_abs_err <= TABULATED_ABS_TOL else "FAIL",
            ],
            [
                "rmsd_ala",
                f"{rmsd_expected:.6f}",
                f"{rmsd_actual:.6f}",
                f"{rmsd_abs_err:.6f}",
                f"{RMSD_ABS_TOL:.6f}",
                "PASS" if rmsd_abs_err <= RMSD_ABS_TOL else "FAIL",
            ],
        ],
        title=f"CV Validation: {CASE_NAME}",
    )

    assert distance_abs_err <= DISTANCE_ABS_TOL
    assert angle_abs_err <= ANGLE_ABS_TOL
    assert phi_abs_err <= DIHEDRAL_ABS_TOL
    assert psi_abs_err <= DIHEDRAL_ABS_TOL
    assert combo_abs_err <= COMBINATION_ABS_TOL
    assert tabulated_abs_err <= TABULATED_ABS_TOL
    assert rmsd_abs_err <= RMSD_ABS_TOL
