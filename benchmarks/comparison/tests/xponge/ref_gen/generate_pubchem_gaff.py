#!/usr/bin/env python3

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils import Runner

PUBCHEM_SDF_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF"
    "?record_type=3d"
)


def get_repo_root() -> Path:
    return REPO_ROOT


def require_executables(*names):
    missing = []
    for name in names:
        if shutil.which(name) is None:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Required AmberTools executable(s) are missing from PATH: "
            + ", ".join(missing)
        )


def enabled_reference_tools(args):
    tools = ["antechamber"]
    if not args.skip_frcmod:
        tools.append("parmchk2")
    if not args.skip_energy:
        tools.extend(["tleap", "sander"])
    return tools


def download_pubchem_sdf(cid: int, output_path: Path, timeout: float):
    url = PUBCHEM_SDF_URL.format(cid=cid)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SPONGE-xponge-benchmark/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 404, 503}:
            return False, f"PubChem HTTP {exc.code}"
        raise
    if not data or b"$$$$" not in data:
        return False, "PubChem response is not an SDF record"
    output_path.write_bytes(data)
    return True, "ok"


def run_antechamber(
    work_dir: Path,
    input_path: Path,
    output_mol2_path: Path,
    *,
    input_format: str,
    atom_type: str,
    timeout: float,
):
    Runner.run_command(
        [
            "antechamber",
            "-i", input_path.name,
            "-fi", input_format,
            "-o", output_mol2_path.name,
            "-fo", "mol2",
            "-at", atom_type,
            "-c", "dc",
            "-s", "2",
            "-pf", "y",
        ],
        cwd=work_dir,
        timeout=timeout,
    )


def run_parmchk2_to_frcmod(
    work_dir: Path,
    reference_mol2_path: Path,
    frcmod_path: Path,
    timeout: float,
):
    Runner.run_command(
        [
            "parmchk2",
            "-i",
            reference_mol2_path.name,
            "-f",
            "mol2",
            "-o",
            frcmod_path.name,
            "-s",
            "gaff",
        ],
        cwd=work_dir,
        timeout=timeout,
    )


def _write_amber_single_point_inputs(work_dir: Path):
    (work_dir / "tleap.in").write_text(
        "\n".join(
            [
                "source leaprc.gaff",
                "loadamberparams ligand.frcmod",
                "MOL = loadmol2 ligand.mol2",
                "check MOL",
                "saveamberparm MOL system.parm7 system.rst7",
                "quit",
                "",
            ]
        )
    )
    (work_dir / "sander.in").write_text(
        "\n".join(
            [
                "Run0 xponge GAFF gas-phase single point",
                "&cntrl",
                "  imin=0, irest=0, ntx=1,",
                "  nstlim=0, dt=0.001,",
                "  ntb=0, igb=0, cut=999.0,",
                "  ntc=1, ntf=1,",
                "  ntpr=1, ntwx=0, ntwr=0,",
                "/",
                "",
            ]
        )
    )


SANDER_ENERGY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9-])"
    r"(BOND|ANGLE|DIHED|VDWAALS|EELEC|1-4\s+(?:NB|VDW)|1-4\s+EEL|EPtot)"
    r"\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

SANDER_ENERGY_KEYS = {
    "BOND": "bond",
    "ANGLE": "angle",
    "DIHED": "dihedral",
    "VDWAALS": "vdw",
    "EELEC": "elec",
    "1-4NB": "one_four_vdw",
    "1-4VDW": "one_four_vdw",
    "1-4EEL": "one_four_elec",
    "EPtot": "total",
}


def parse_sander_energy_terms(sander_out_path: Path):
    text = sander_out_path.read_text()
    terms = {}
    for label, value in SANDER_ENERGY_PATTERN.findall(text):
        key = SANDER_ENERGY_KEYS["".join(label.split())]
        terms[key] = float(value)
    missing = sorted(set(SANDER_ENERGY_KEYS.values()) - set(terms))
    if missing:
        raise ValueError(
            f"Failed to parse Amber energy term(s) {missing} from {sander_out_path}"
        )
    return terms


def run_amber_gaff_single_point_energy(
    reference_mol2_path: Path, frcmod_path: Path, timeout: float
):
    with tempfile.TemporaryDirectory(prefix="xponge_pubchem_energy_") as tmp:
        work_dir = Path(tmp)
        shutil.copy2(reference_mol2_path, work_dir / "ligand.mol2")
        shutil.copy2(frcmod_path, work_dir / "ligand.frcmod")
        _write_amber_single_point_inputs(work_dir)
        Runner.run_command(["tleap", "-f", "tleap.in"], cwd=work_dir, timeout=timeout)
        Runner.run_command(
            [
                "sander",
                "-O",
                "-i",
                "sander.in",
                "-o",
                "sander.out",
                "-p",
                "system.parm7",
                "-c",
                "system.rst7",
                "-r",
                "system_out.rst7",
                "-inf",
                "system.mdinfo",
            ],
            cwd=work_dir,
            timeout=timeout,
        )
        return parse_sander_energy_terms(work_dir / "sander.out")


def parse_mol2_summary(mol2_path: Path):
    atom_types = []
    atom_charges = []
    bonds = []
    section = None

    for line in mol2_path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line[len("@<TRIPOS>") :].strip()
            continue
        if not line.strip():
            continue
        words = line.split()
        if section == "ATOM":
            atom_types.append(words[5])
            atom_charges.append(float(words[8]))
        elif section == "BOND":
            bonds.append([int(words[1]), int(words[2]), words[3]])

    if not atom_types:
        raise ValueError(f"No atoms parsed from {mol2_path}")
    return {
        "atom_count": len(atom_types),
        "bond_count": len(bonds),
        "gaff_atom_types": atom_types,
        "charge_sum": round(sum(atom_charges), 6),
        "bonds": bonds,
    }


def is_gaff_suitable(summary, max_atoms):
    if summary["atom_count"] <= 0 or summary["atom_count"] > max_atoms:
        return False
    if summary["bond_count"] <= 0:
        return False
    for atom_type in summary["gaff_atom_types"]:
        element = "".join(ch for ch in atom_type if ch.isalpha()).lower()
        if element and element[0] not in {
            "h",
            "c",
            "n",
            "o",
            "s",
            "p",
            "f",
            "i",
            "b",
        }:
            return False
    return True


def build_reference_entry(
    output_root: Path,
    case_name: str,
    cid: int,
    input_mol2_path: Path,
    reference_mol2_path: Path,
    reference_frcmod_path: Path,
    args,
    existing_entry=None,
    summary=None,
):
    if summary is None:
        summary = parse_mol2_summary(reference_mol2_path)
    if not args.skip_frcmod and (
        args.refresh_frcmod or not reference_frcmod_path.exists()
    ):
        with tempfile.TemporaryDirectory(prefix="xponge_pubchem_frcmod_") as tmp:
            work_dir = Path(tmp)
            work_mol2_path = work_dir / reference_mol2_path.name
            work_frcmod_path = work_dir / reference_frcmod_path.name
            shutil.copy2(reference_mol2_path, work_mol2_path)
            run_parmchk2_to_frcmod(
                work_dir, work_mol2_path, work_frcmod_path, args.energy_timeout
            )
            shutil.copy2(work_frcmod_path, reference_frcmod_path)
    entry = {
        "case_name": case_name,
        "cid": cid,
        "input_mol2_file": str(input_mol2_path.relative_to(output_root)),
        "reference_mol2_file": str(reference_mol2_path.relative_to(output_root)),
        **summary,
    }
    if reference_frcmod_path.exists():
        entry["reference_frcmod_file"] = str(
            reference_frcmod_path.relative_to(output_root)
        )
    if args.skip_energy:
        return entry
    if (
        existing_entry
        and "amber_energy" in existing_entry
        and not args.refresh_energy
    ):
        entry["amber_energy"] = existing_entry["amber_energy"]
    else:
        entry["amber_energy"] = run_amber_gaff_single_point_energy(
            reference_mol2_path, reference_frcmod_path, timeout=args.energy_timeout
        )
    return entry


def load_existing_entries(reference_path: Path):
    if not reference_path.exists():
        return []
    payload = json.loads(reference_path.read_text())
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    return entries


def generate_reference(args):
    if args.skip_frcmod and not args.skip_energy:
        raise ValueError("--skip-frcmod requires --skip-energy")

    rng = random.Random(args.seed)
    output_root = args.output_root.resolve()
    input_dir = output_root / "input_mol2"
    reference_dir = output_root / "reference_mol2"
    frcmod_dir = output_root / "reference_frcmod"
    for directory in (input_dir, reference_dir, frcmod_dir):
        directory.mkdir(parents=True, exist_ok=True)
    reference_path = output_root / "reference.json"

    entries = []
    failures = []
    existing_entries = load_existing_entries(reference_path)
    if len(existing_entries) < args.count:
        require_executables("antechamber")
    if not args.skip_frcmod:
        require_executables("parmchk2")
    if not args.skip_energy:
        require_executables("tleap", "sander")

    if len(existing_entries) >= args.count:
        for existing_entry in existing_entries[: args.count]:
            case_name = existing_entry["case_name"]
            final_input_mol2 = output_root / existing_entry["input_mol2_file"]
            final_reference_mol2 = output_root / existing_entry["reference_mol2_file"]
            final_reference_frcmod = frcmod_dir / f"{case_name}.frcmod"
            if "reference_frcmod_file" in existing_entry:
                final_reference_frcmod = output_root / existing_entry[
                    "reference_frcmod_file"
                ]
            try:
                entries.append(
                    build_reference_entry(
                        output_root,
                        case_name,
                        int(existing_entry["cid"]),
                        final_input_mol2,
                        final_reference_mol2,
                        final_reference_frcmod,
                        args,
                        existing_entry=existing_entry,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - generator records failed cases.
                failures.append(
                    {
                        "cid": int(existing_entry["cid"]),
                        "stage": "amber_energy",
                        "detail": str(exc),
                    }
                )
            if (
                len(entries) % args.progress_interval == 0
                or len(entries) == args.count
            ):
                print(f"[xponge-ref] refreshed={len(entries)}")
        attempts = 0
        seen_cids = {int(entry["cid"]) for entry in existing_entries}
    else:
        attempts = 0
        seen_cids = set()

    while len(entries) < args.count and attempts < args.max_attempts:
        attempts += 1
        cid = rng.randint(args.min_cid, args.max_cid)
        if cid in seen_cids:
            continue
        seen_cids.add(cid)

        case_name = f"cid_{cid}"
        final_input_mol2 = input_dir / f"{case_name}.mol2"
        final_reference_mol2 = reference_dir / f"{case_name}.mol2"
        final_reference_frcmod = frcmod_dir / f"{case_name}.frcmod"

        if final_input_mol2.exists() and final_reference_mol2.exists():
            entries.append(
                build_reference_entry(
                    output_root,
                    case_name,
                    cid,
                    final_input_mol2,
                    final_reference_mol2,
                    final_reference_frcmod,
                    args,
                ),
            )
            continue

        try:
            with tempfile.TemporaryDirectory(prefix="xponge_pubchem_gaff_") as tmp:
                work_dir = Path(tmp)
                sdf_path = work_dir / f"{case_name}.sdf"
                input_mol2_path = work_dir / f"{case_name}.input.mol2"
                reference_mol2_path = work_dir / f"{case_name}.gaff.mol2"

                ok, detail = download_pubchem_sdf(
                    cid, sdf_path, timeout=args.download_timeout
                )
                if not ok:
                    failures.append({"cid": cid, "stage": "pubchem", "detail": detail})
                    continue

                run_antechamber(
                    work_dir,
                    sdf_path,
                    input_mol2_path,
                    input_format="sdf",
                    atom_type="sybyl",
                    timeout=args.command_timeout,
                )
                run_antechamber(
                    work_dir,
                    input_mol2_path,
                    reference_mol2_path,
                    input_format="mol2",
                    atom_type="gaff",
                    timeout=args.command_timeout,
                )

                summary = parse_mol2_summary(reference_mol2_path)
                if not is_gaff_suitable(summary, args.max_atoms):
                    failures.append(
                        {
                            "cid": cid,
                            "stage": "filter",
                            "detail": "not suitable for this GAFF benchmark",
                        }
                    )
                    continue

                shutil.copy2(input_mol2_path, final_input_mol2)
                shutil.copy2(reference_mol2_path, final_reference_mol2)
                entries.append(
                    build_reference_entry(
                        output_root,
                        case_name,
                        cid,
                        final_input_mol2,
                        final_reference_mol2,
                        final_reference_frcmod,
                        args,
                        summary=summary,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - generator records failed candidates.
            failures.append({"cid": cid, "stage": "ambertools", "detail": str(exc)})

        if attempts % args.progress_interval == 0 or len(entries) == args.count:
            print(
                f"[xponge-ref] attempts={attempts} kept={len(entries)} "
                f"failed={len(failures)}"
            )
        if args.request_delay > 0:
            time.sleep(args.request_delay)

    if len(entries) < args.count:
        raise RuntimeError(
            f"Only generated {len(entries)} successful cases after {attempts} attempts"
        )

    payload = {
        "metadata": {
            "suite": "xponge_pubchem_gaff",
            "generator": (
                "benchmarks/comparison/tests/xponge/ref_gen/"
                "generate_pubchem_gaff.py"
            ),
            "seed": args.seed,
            "target_count": args.count,
            "attempts": attempts,
            "min_cid": args.min_cid,
            "max_cid": args.max_cid,
            "max_atoms": args.max_atoms,
            "reference_tools": enabled_reference_tools(args),
        },
        "entries": entries,
        "failures": failures[-args.keep_failures :],
    }
    reference_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[OK] Wrote {len(entries)} xponge GAFF entries to {reference_path}")


def main():
    repo_root = get_repo_root()
    default_output_root = (
        repo_root
        / "benchmarks"
        / "comparison"
        / "tests"
        / "xponge"
        / "statics"
        / "pubchem_gaff_1000"
    )
    parser = argparse.ArgumentParser(
        description="Generate AmberTools GAFF references for random PubChem molecules."
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--min-cid", type=int, default=1)
    parser.add_argument("--max-cid", type=int, default=120_000_000)
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=50_000)
    parser.add_argument("--download-timeout", type=float, default=20.0)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--energy-timeout", type=float, default=120.0)
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--skip-frcmod", action="store_true")
    parser.add_argument("--refresh-energy", action="store_true")
    parser.add_argument("--refresh-frcmod", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--keep-failures", type=int, default=200)
    args = parser.parse_args()
    generate_reference(args)


if __name__ == "__main__":
    main()
