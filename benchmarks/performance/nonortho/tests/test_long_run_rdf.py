from pathlib import Path

import numpy as np

from benchmarks.utils import Outputer
from benchmarks.validation.thermostat.tests.utils import read_mass_values

from benchmarks.performance.nonortho.tests.utils import (
    compute_oo_rdf,
    read_atom_names,
    read_box_trajectory,
    read_coordinate_atom_count,
    read_coordinate_trajectory,
    run_sponge_barostat,
    save_rdf_plot,
    summarize_rdf,
    write_velocity_file_with_zero_mass_support,
    write_nonortho_long_run_mdin,
)


def test_wat_nonortho_long_run_oo_rdf(
    statics_path,
    outputs_path,
    nonortho_mode,
    nonortho_steps,
    mpi_np,
):
    case_name = "WAT_nonortho"
    run_name = f"{nonortho_mode.lower()}_long_run_rdf"
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        mpi_np=mpi_np,
        run_name=run_name,
    )

    masses = read_mass_values(case_dir / "WAT_mass.txt")
    velocity_file = case_dir / "initial_velocity.txt"
    write_velocity_file_with_zero_mass_support(
        velocity_file,
        masses,
        temperature=300.0,
        seed=2026,
    )

    trajectory_interval = max(100, nonortho_steps // 50)
    write_nonortho_long_run_mdin(
        case_dir,
        mode=nonortho_mode,
        step_limit=nonortho_steps,
        trajectory_interval=trajectory_interval,
        velocity_in_file=velocity_file.name,
    )

    timeout = max(2400, nonortho_steps // 20)
    run_sponge_barostat(case_dir, timeout=timeout, mpi_np=mpi_np)

    atom_count = read_coordinate_atom_count(case_dir / "WAT_coordinate.txt")
    atom_names = read_atom_names(case_dir / "WAT_atom_name.txt")
    oxygen_indices = np.asarray(
        [idx for idx, name in enumerate(atom_names) if name == "O"], dtype=int
    )
    trajectory = read_coordinate_trajectory(case_dir / "mdcrd.dat", atom_count)
    box_trajectory = read_box_trajectory(case_dir / "mdbox.txt")
    radii, rdf = compute_oo_rdf(
        trajectory,
        box_trajectory,
        oxygen_indices,
        bin_width=0.05,
    )
    rdf_summary = summarize_rdf(radii, rdf)
    rdf_plot = save_rdf_plot(
        radii,
        rdf,
        case_dir / "wat_nonortho_oo_rdf.png",
        title=f"WAT_nonortho O-O RDF ({nonortho_mode}, {nonortho_steps} steps)",
    )

    frame_count = min(trajectory.shape[0], box_trajectory.shape[0])
    rows = [
        ["RunName", run_name],
        ["Mode", nonortho_mode],
        ["StepLimit", str(nonortho_steps)],
        ["TrajectoryInterval", str(trajectory_interval)],
        ["FramesUsed", str(frame_count)],
        ["OxygenCount", str(len(oxygen_indices))],
        ["RDFPeakR(A)", f"{rdf_summary['rdf_peak_r']:.3f}"],
        ["RDFPeakG", f"{rdf_summary['rdf_peak_g']:.3f}"],
        ["RDFMax", f"{rdf_summary['rdf_max']:.3f}"],
        ["RDFTailMean", f"{rdf_summary['rdf_tail_mean']:.3f}"],
        [
            "RDFPlot",
            rdf_plot.name if rdf_plot is not None else "not-generated",
        ],
        ["Result", "PLOTTED"],
    ]
    Outputer.print_table(
        ["Metric", "Value"],
        rows,
        title=f"Nonortho Performance: Long {nonortho_mode} O-O RDF",
    )

    assert frame_count >= 2
    assert np.all(np.isfinite(rdf))
