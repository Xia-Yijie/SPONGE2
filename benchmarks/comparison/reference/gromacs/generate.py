#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CASE_NAME = "alanine_dipeptide_charmm_tip3p"
PERTURB_CASES = [
    (0, 0.00),
    (1, 0.02),
    (2, 0.05),
]
TERM_KEYS = [
    "bond",
    "angle",
    "urey_bradley",
    "proper_dihedral",
    "improper_dihedral",
    "lj14",
    "coulomb14",
    "lj_sr",
    "coulomb_sr",
    "coulomb_recip",
    "potential",
    "pressure",
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_gromacs_utils():
    tests_dir = (
        get_repo_root()
        / "benchmarks"
        / "comparison"
        / "tests"
        / "gromacs"
        / "tests"
    )
    sys.path.insert(0, str(tests_dir))

    import utils as gromacs_utils

    return gromacs_utils


def _iter_case_specs():
    for iteration, perturbation in PERTURB_CASES:
        yield {
            "case_name": CASE_NAME,
            "iteration": int(iteration),
            "perturbation": float(perturbation),
            "seed": int(20260217 + 1000 * iteration),
        }


def _detect_gromacs_version():
    result = subprocess.run(
        ["gmx", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to query GROMACS version.\nOutput tail:\n{output[-2000:]}"
        )
    for line in output.splitlines():
        if "GROMACS version" in line:
            return line.split(":", 1)[-1].strip()
    return "unknown"


def _copy_reference_tree(src_root: Path, dst_root: Path, rel_name: str):
    src = src_root / rel_name
    dst = dst_root / rel_name
    if dst.exists():
        shutil.rmtree(dst)
    if not src.exists():
        raise FileNotFoundError(f"Missing GROMACS reference directory at {src}")
    shutil.copytree(src, dst)


def _copy_generated_sponge_inputs(case_dir: Path, dst_dir: Path):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(Path(case_dir).glob("sys_flexible*")):
        if not src.is_file():
            continue
        shutil.copy2(src, dst_dir / src.name)
        copied += 1
    if copied == 0:
        raise ValueError(
            f"No generated SPONGE input files (sys_flexible*) found in {case_dir}"
        )


def generate_payload(statics_path: Path, reference_root: Path):
    gromacs_utils = _load_gromacs_utils()
    gromacs_utils.require_gromacs()
    gromacs_utils.require_xponge()

    statics_root = reference_root / "statics"
    forces_root = reference_root / "forces"
    sponge_inputs_root = reference_root / "sponge_inputs"
    if not statics_root.exists():
        raise FileNotFoundError(
            f"Missing GROMACS reference statics directory: {statics_root}"
        )
    if forces_root.exists():
        shutil.rmtree(forces_root)
    forces_root.mkdir(parents=True, exist_ok=True)
    if sponge_inputs_root.exists():
        shutil.rmtree(sponge_inputs_root)
    sponge_inputs_root.mkdir(parents=True, exist_ok=True)

    entries = []
    with tempfile.TemporaryDirectory(
        prefix="gromacs_refgen_outputs_"
    ) as tmpdir:
        outputs_path = Path(tmpdir)
        for spec in _iter_case_specs():
            case_name = spec["case_name"]
            iteration = spec["iteration"]
            perturbation = spec["perturbation"]
            seed = spec["seed"]
            run_tag = f"_refgen/{case_name}/{iteration}"

            case_dir = gromacs_utils.prepare_output_case(
                statics_path=statics_path,
                outputs_path=outputs_path,
                case_name=case_name,
                run_tag=run_tag,
            )
            gromacs_utils.copy_gromacs_reference_case_files(
                statics_path=statics_path,
                case_dir=case_dir,
                case_name=case_name,
            )
            gromacs_utils.link_charmm27_forcefield(case_dir)
            gromacs_utils.perturb_gro_inplace(
                case_dir / "solv.gro",
                perturbation_angstrom=perturbation,
                seed=seed,
                perturb_non_water=False,
            )
            gromacs_utils.generate_sponge_inputs_from_gromacs(
                case_dir,
                output_prefix="sys_flexible",
            )
            sponge_rel_dir = (
                Path("sponge_inputs") / case_name / f"iter{iteration}"
            )
            _copy_generated_sponge_inputs(
                case_dir=case_dir,
                dst_dir=reference_root / sponge_rel_dir,
            )

            gromacs_utils.run_gromacs_flexible_run0(case_dir)
            gmx_terms = gromacs_utils.extract_gromacs_terms(case_dir)
            gmx_forces = gromacs_utils.extract_gromacs_forces(case_dir)

            force_rel_path = Path("forces") / case_name / f"iter{iteration}.npy"
            force_path = reference_root / force_rel_path
            force_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(force_path, gmx_forces.astype(np.float64))

            entry = {
                "case_name": case_name,
                "iteration": iteration,
                "perturbation": perturbation,
                "seed": seed,
                "natom": int(gmx_forces.shape[0]),
                "forces_file": str(force_rel_path).replace("\\", "/"),
                "sponge_inputs_dir": str(sponge_rel_dir).replace("\\", "/"),
            }
            for key in TERM_KEYS:
                entry[key] = float(gmx_terms[key])
            entries.append(entry)

    entries.sort(key=lambda v: (v["case_name"], v["iteration"]))
    return {
        "format_version": 1,
        "energy_unit": "kcal/mol",
        "force_unit": "kcal/mol/Angstrom",
        "pressure_unit": "bar",
        "gromacs_version": _detect_gromacs_version(),
        "entries": entries,
    }


def _entries_to_map(payload):
    result = {}
    entries = payload.get("entries", [])
    for entry in entries:
        key = (entry["case_name"], int(entry["iteration"]))
        result[key] = entry
    return result


def _list_relative_files(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Missing directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return files


def _compare_file_trees(current_root: Path, generated_root: Path):
    current_files = _list_relative_files(current_root)
    generated_files = _list_relative_files(generated_root)
    if current_files != generated_files:
        return (
            False,
            f"File set differs: current={len(current_files)}, "
            f"generated={len(generated_files)}",
        )
    return True, "ok"


def compare_payloads(
    current_payload,
    generated_payload,
    current_root: Path,
    generated_root: Path,
    abs_tol_term: float,
    abs_tol_force: float,
):
    for key in [
        "format_version",
        "energy_unit",
        "force_unit",
        "pressure_unit",
        "gromacs_version",
    ]:
        if current_payload.get(key) != generated_payload.get(key):
            return (
                False,
                f"Metadata differs at '{key}': current={current_payload.get(key)!r}, "
                f"generated={generated_payload.get(key)!r}",
            )

    current_map = _entries_to_map(current_payload)
    generated_map = _entries_to_map(generated_payload)
    if set(current_map) != set(generated_map):
        current_only = sorted(set(current_map) - set(generated_map))
        generated_only = sorted(set(generated_map) - set(current_map))
        return (
            False,
            "Entry keys differ. "
            f"current-only={len(current_only)}, generated-only={len(generated_only)}",
        )

    max_term_diff = 0.0
    max_term_key = None
    max_force_diff = 0.0
    max_force_key = None
    max_term_name = None

    for key in sorted(current_map):
        current_entry = current_map[key]
        generated_entry = generated_map[key]
        if (
            abs(
                float(current_entry["perturbation"])
                - float(generated_entry["perturbation"])
            )
            > 1.0e-12
        ):
            return (
                False,
                f"Perturbation differs for {key}: current={current_entry['perturbation']}, "
                f"generated={generated_entry['perturbation']}",
            )
        if int(current_entry["seed"]) != int(generated_entry["seed"]):
            return (
                False,
                f"Seed differs for {key}: current={current_entry['seed']}, "
                f"generated={generated_entry['seed']}",
            )
        if int(current_entry["natom"]) != int(generated_entry["natom"]):
            return (
                False,
                f"natom differs for {key}: current={current_entry['natom']}, "
                f"generated={generated_entry['natom']}",
            )
        current_sponge_rel_dir = current_entry.get("sponge_inputs_dir")
        generated_sponge_rel_dir = generated_entry.get("sponge_inputs_dir")
        if (
            not isinstance(current_sponge_rel_dir, str)
            or not current_sponge_rel_dir
            or not isinstance(generated_sponge_rel_dir, str)
            or not generated_sponge_rel_dir
        ):
            return (
                False,
                f"Missing sponge_inputs_dir metadata for {key}",
            )
        if current_sponge_rel_dir != generated_sponge_rel_dir:
            return (
                False,
                f"sponge_inputs_dir differs for {key}: "
                f"current={current_sponge_rel_dir}, generated={generated_sponge_rel_dir}",
            )

        for term_name in TERM_KEYS:
            term_diff = abs(
                float(current_entry[term_name])
                - float(generated_entry[term_name])
            )
            if term_diff > max_term_diff:
                max_term_diff = term_diff
                max_term_key = key
                max_term_name = term_name

        current_force = np.asarray(
            np.load(current_root / current_entry["forces_file"]),
            dtype=np.float64,
        )
        generated_force = np.asarray(
            np.load(generated_root / generated_entry["forces_file"]),
            dtype=np.float64,
        )
        if current_force.shape != generated_force.shape:
            return (
                False,
                f"Force shape differs for {key}: current={current_force.shape}, "
                f"generated={generated_force.shape}",
            )
        force_diff = float(np.max(np.abs(current_force - generated_force)))
        if force_diff > max_force_diff:
            max_force_diff = force_diff
            max_force_key = key
        same_files, files_detail = _compare_file_trees(
            current_root / current_sponge_rel_dir,
            generated_root / generated_sponge_rel_dir,
        )
        if not same_files:
            return (
                False,
                f"SPONGE input files differ for {key}. {files_detail}",
            )

    if max_term_diff > abs_tol_term:
        return (
            False,
            f"Term differs above tolerance: max_diff={max_term_diff:.3e} "
            f"at {max_term_key}, term={max_term_name}",
        )
    if max_force_diff > abs_tol_force:
        return (
            False,
            f"Force differs above tolerance: max_diff={max_force_diff:.3e} at {max_force_key}",
        )

    return (
        True,
        f"max_term_diff={max_term_diff:.3e}, max_force_diff={max_force_diff:.3e}",
    )


def main():
    repo_root = get_repo_root()
    default_output = (
        repo_root
        / "benchmarks"
        / "comparison"
        / "reference"
        / "gromacs"
        / "reference.json"
    )
    default_statics = (
        repo_root
        / "benchmarks"
        / "comparison"
        / "tests"
        / "gromacs"
        / "statics"
    )

    parser = argparse.ArgumentParser(
        description="Generate static GROMACS reference energies/forces for comp-gromacs tests."
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--statics-path", type=Path, default=default_statics)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--abs-tol-term", type=float, default=1.0e-6)
    parser.add_argument("--abs-tol-force", type=float, default=1.0e-6)
    args = parser.parse_args()

    if args.check:
        if not args.output.exists():
            print(f"[FAIL] Missing reference file: {args.output}")
            raise SystemExit(1)
        with tempfile.TemporaryDirectory(
            prefix="gromacs_refgen_check_"
        ) as tmpdir:
            tmp_output = Path(tmpdir) / "reference.json"
            tmp_root = tmp_output.parent
            _copy_reference_tree(
                args.output.parent.resolve(),
                tmp_root,
                rel_name="statics",
            )
            _copy_reference_tree(
                args.output.parent.resolve(),
                tmp_root,
                rel_name="sponge_inputs",
            )
            generated_payload = generate_payload(args.statics_path, tmp_root)
            tmp_output.write_text(
                json.dumps(generated_payload, indent=2) + "\n"
            )

            current_payload = json.loads(args.output.read_text())
            ok, detail = compare_payloads(
                current_payload=current_payload,
                generated_payload=generated_payload,
                current_root=args.output.parent,
                generated_root=tmp_root,
                abs_tol_term=args.abs_tol_term,
                abs_tol_force=args.abs_tol_force,
            )
            if ok:
                print(f"[PASS] GROMACS reference is up to date ({detail}).")
                return
            print(f"[FAIL] GROMACS reference differs. {detail}")
            raise SystemExit(1)

    payload = generate_payload(args.statics_path, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[OK] Wrote {len(payload['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
