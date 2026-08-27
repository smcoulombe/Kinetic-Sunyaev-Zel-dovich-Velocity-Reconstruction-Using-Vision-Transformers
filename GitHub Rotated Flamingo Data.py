"""
Generate rotated versions of the FLAMINGO real-map fields by applying fixed
Euler-angle rotations, and save each rotated map to a new file.

Source maps (flamingo_real_{name}_nside{NSIDE}.npy) are treated as read-only
inputs: this script never opens them in write mode and the source paths never
appear on the left-hand side of any write. Rotated maps are written under
brand-new filenames that encode the rotation, so nothing that already exists
on disk is ever touched -- only new files get created.

Designed to be submitted as a batch job (see accompanying .sbatch script)
rather than run interactively.
"""
import os
import argparse
import shutil
import numpy as np
import healpy as hp
import gc
from multiprocessing import Pool

DATA_DIR = "/mnt/beegfs/scoulombe"
NSIDE = 2048
npix = hp.nside2npix(NSIDE)

# --- Source maps (READ-ONLY inputs; must already exist) ---
FIELD_NAMES = ["T", "v0", "v1", "v2", "tau0", "tau1", "tau2", "cmbnoise"]
SOURCE_PATHS = {
    name: os.path.join(DATA_DIR, f"flamingo_real_{name}_nside{NSIDE}.npy")
    for name in FIELD_NAMES
}

# --- Rotations to apply, as (angle1, angle2, angle3) in degrees. ---
# NOTE: healpy.Rotator interprets these as Euler angles under its default
# `eulertype` ("ZYX") since none is passed explicitly here -- matching the
# rotate_map_pair(rot=rot_angles, deg=True) example.
ROTATIONS = [
    (45, 45, 0),
    (45, 135, 0),
    (135, 45, 0),
    (135, 135, 0),
]


def _rot_tag(rot):
    a, b, c = rot
    return f"rot{a}_{b}_{c}"


def _out_path(name, rot):
    return os.path.join(
        DATA_DIR, f"flamingo_real_{name}_nside{NSIDE}_{_rot_tag(rot)}.npy"
    )


def _all_outputs():
    """Every (field, rotation) -> final output path."""
    return {
        (name, rot): _out_path(name, rot)
        for name in FIELD_NAMES
        for rot in ROTATIONS
    }


def load_source(name):
    path = SOURCE_PATHS[name]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- source maps must already exist before rotating."
        )
    arr = np.load(path)
    if arr.shape != (npix,):
        raise ValueError(f"{path} has shape {arr.shape}, expected ({npix},)")
    return arr


def rotate_one(args):
    """Rotate one (field, rotation) pair purely in memory and return it.
    No file I/O happens in worker processes at all, avoiding many workers
    concurrently writing/renaming on the network filesystem.
    """
    name, rot, map_in = args
    rotator = hp.Rotator(rot=rot, deg=True)
    rotated = rotator.rotate_map_pixel(map_in)
    return name, rot, rotated.astype(np.float32)


def main(n_workers):
    all_outputs = _all_outputs()

    missing = {k: p for k, p in all_outputs.items() if not os.path.exists(p)}
    if not missing:
        print("Found all rotated map files already, skipping generation.")
        return

    print(
        f"{len(all_outputs) - len(missing)}/{len(all_outputs)} rotated files "
        f"already exist -- generating the remaining {len(missing)}."
    )

    # Only load the source fields we actually still need, once each.
    needed_names = sorted({name for (name, rot) in missing})
    sources = {name: load_source(name) for name in needed_names}

    tasks = [(name, rot, sources[name]) for (name, rot) in missing]

    print(f"Rotating {len(tasks)} (field, rotation) pairs using {n_workers} worker processes...")
    with Pool(n_workers) as pool:
        for name, rot, rotated in pool.imap_unordered(rotate_one, tasks):
            final_path = all_outputs[(name, rot)]
            tmp_path = final_path + ".tmp.npy"

            # Guard again right before writing: if another process already
            # finished this exact file since we built `missing`, skip it
            # rather than clobbering it.
            if os.path.exists(final_path):
                print(f"Skipping {name} {_rot_tag(rot)} -- already exists (written concurrently).")
                del rotated
                gc.collect()
                continue

            # Write to a temp filename first; only rename to the final name
            # once fully written and flushed. A job killed mid-write leaves
            # only an orphaned .tmp.npy, never a half-written "final" file,
            # so reruns are always safe and existing outputs are never
            # touched until their replacement is completely ready.
            np.save(tmp_path, rotated)
            shutil.move(tmp_path, final_path)
            print(f"Saved {name} {_rot_tag(rot)} -> {final_path}", flush=True)
            del rotated
            gc.collect()

    del sources
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_workers", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)),
        help="Number of parallel worker processes (defaults to SLURM_CPUS_PER_TASK, or CPU count)",
    )
    args = parser.parse_args()
    main(args.n_workers)