#!/usr/bin/env python3

import argparse
import json
import random
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


def require_ambertools():
    missing = []
    for exe in ("antechamber",):
        if shutil.which(exe) is None:
            missing.append(exe)
    if missing:
        raise RuntimeError(
            "Required AmberTools executable(s) are missing from PATH: "
            + ", ".join(missing)
        )


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


def run_antechamber_to_input_mol2(
    work_dir: Path, sdf_path: Path, mol2_path: Path, timeout: float
):
    Runner.run_command(
        [
            "antechamber",
            "-i",
            sdf_path.name,
            "-fi",
            "sdf",
            "-o",
            mol2_path.name,
            "-fo",
            "mol2",
            "-at",
            "sybyl",
            "-c",
            "dc",
            "-s",
            "2",
            "-pf",
            "y",
        ],
        cwd=work_dir,
        timeout=timeout,
    )


def run_antechamber_to_gaff_mol2(
    work_dir: Path,
    input_mol2_path: Path,
    reference_mol2_path: Path,
    timeout: float,
):
    Runner.run_command(
        [
            "antechamber",
            "-i",
            input_mol2_path.name,
            "-fi",
            "mol2",
            "-o",
            reference_mol2_path.name,
            "-fo",
            "mol2",
            "-at",
            "gaff",
            "-c",
            "dc",
            "-s",
            "2",
            "-pf",
            "y",
        ],
        cwd=work_dir,
        timeout=timeout,
    )


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


def generate_reference(args):
    require_ambertools()

    rng = random.Random(args.seed)
    output_root = args.output_root.resolve()
    input_dir = output_root / "input_mol2"
    reference_dir = output_root / "reference_mol2"
    for directory in (input_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    entries = []
    failures = []
    seen_cids = set()
    attempts = 0

    while len(entries) < args.count and attempts < args.max_attempts:
        attempts += 1
        cid = rng.randint(args.min_cid, args.max_cid)
        if cid in seen_cids:
            continue
        seen_cids.add(cid)

        case_name = f"cid_{cid}"
        final_input_mol2 = input_dir / f"{case_name}.mol2"
        final_reference_mol2 = reference_dir / f"{case_name}.mol2"

        if final_input_mol2.exists() and final_reference_mol2.exists():
            summary = parse_mol2_summary(final_reference_mol2)
            entries.append(
                {
                    "case_name": case_name,
                    "cid": cid,
                    "input_mol2_file": str(
                        final_input_mol2.relative_to(output_root)
                    ),
                    "reference_mol2_file": str(
                        final_reference_mol2.relative_to(output_root)
                    ),
                    **summary,
                }
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

                run_antechamber_to_input_mol2(
                    work_dir,
                    sdf_path,
                    input_mol2_path,
                    timeout=args.command_timeout,
                )
                run_antechamber_to_gaff_mol2(
                    work_dir,
                    input_mol2_path,
                    reference_mol2_path,
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
                    {
                        "case_name": case_name,
                        "cid": cid,
                        "input_mol2_file": str(
                            final_input_mol2.relative_to(output_root)
                        ),
                        "reference_mol2_file": str(
                            final_reference_mol2.relative_to(output_root)
                        ),
                        **summary,
                    }
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
            "reference_tools": ["antechamber"],
        },
        "entries": entries,
        "failures": failures[-args.keep_failures :],
    }
    reference_path = output_root / "reference.json"
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
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--keep-failures", type=int, default=200)
    args = parser.parse_args()
    generate_reference(args)


if __name__ == "__main__":
    main()
