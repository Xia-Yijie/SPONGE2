import json

from benchmarks.utils import Outputer, Runner
from benchmarks.validation.utils import parse_mdout_rows


def test_softcore(statics_path, outputs_path, mpi_np):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="softcore",
        mpi_np=mpi_np,
        run_name="softcore",
    )

    Runner.run_sponge(
        case_dir,
        timeout=1800,
        mpi_np=mpi_np,
        mdin_name="mdin.spg.in",
    )

    reference = json.loads(
        (case_dir / "reference" / "energy_reference_np1.json").read_text()
    )["energies"]
    columns = list(reference.keys())
    energies = parse_mdout_rows(
        case_dir / "mdout.txt",
        columns=columns,
        int_columns=("step",),
    )[0]
    comparable = {
        key: abs(float(energies[key]) - float(ref_val))
        for key, ref_val in reference.items()
        if key not in {"step", "time", "temperature"}
        and key in energies
        and isinstance(ref_val, (int, float))
        and isinstance(energies[key], (int, float))
    }
    max_err_key = max(comparable, key=comparable.get) if comparable else "N/A"
    max_abs_error = comparable[max_err_key] if comparable else 0.0
    abs_tol = 2.0

    Outputer.print_table(
        [
            "Case",
            "MPI_NP",
            "MaxErrTerm",
            "MaxAbsErr",
            "AbsTol",
            "Status",
        ],
        [
            [
                "softcore",
                str(mpi_np) if mpi_np is not None else "direct",
                max_err_key,
                f"{max_abs_error:.6f}",
                f"{abs_tol:.3f}",
                "PASS" if max_abs_error <= abs_tol else "FAIL",
            ]
        ],
        title="Misc Validation: Softcore Point Energy",
    )

    assert max_abs_error <= abs_tol
