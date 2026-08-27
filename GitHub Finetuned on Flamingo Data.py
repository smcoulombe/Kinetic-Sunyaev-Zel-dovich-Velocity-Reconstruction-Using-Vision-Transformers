"""
Fine-tuning pipeline: binned multi-shell ViT, initialized from the stage-24
checkpoint of a prior curriculum run, then fine-tuned on a NEW dataset with
a FIXED (non-curriculum) eps mixture.

This is a derivative of the original curriculum-training script. Everything
about the model architecture and loss functions is unchanged (same schema:
4 input channels [Tnoise, tau0, tau1, tau2], 3 output channels [v0, v1, v2]).
What's different:

  1. NO curriculum. Eps sampling weights are FIXED for the whole run, taken
     from the same weight distribution that was active at the end of stage
     24 in the original run (so the training distribution the model sees
     doesn't jump discontinuously at the start of fine-tuning). Edit
     FIXED_EPS_WEIGHTS below if you want a different mixture.
  2. Initializes model weights from SOURCE_CHECKPOINT_FILE
     (model_weights_{SOURCE_SNAPSHOT_TAG}_stage24_end.pt from the original
     run) the FIRST time this script is run. Subsequent runs resume from
     this fine-tune's OWN checkpoint (CHECKPOINT_FILE), not the source
     checkpoint again -- see the loading logic near the model definition.
  3. DATA: real (non-Gaussian) FLAMINGO maps, not synthetic realizations.
     There are only 5 full-sky maps total per field: one un-rotated
     "flamingo_real_{name}_nside2048.npy" plus 4 rotated copies produced by
     the companion rotate-and-save script (rot45_45_0, rot45_135_0,
     rot135_45_0, rot135_135_0). Per the plan:
       - The 4 ROTATED maps are the train/val pool (realization indices
         0-3), split 3 train / 1 val via the existing realization-index
         split logic (VAL_FRACTION_OF_REALIZATIONS).
       - The UN-ROTATED original map is held out entirely as the test set
         (TEST_FILES) -- real data, never touched during train/val.
       - The old camb+synfast synthetic test-realization generator is
         REMOVED (this data isn't Gaussian, so a synthetic test set
         wouldn't be representative) -- test_snapshot() now always
         evaluates against the real held-out original map.
  4. Fresh, separate PATCH_CACHE_TAG so nothing here touches the original
     curriculum run's patch cache.
  5. Training loop has no stage-transition logic: it's a single phase with
     early stopping (patience) that just stops when it plateaus.
  6. BATCH_SIZE lowered (64, from 256) since the train patch pool is much
     smaller now (3 realizations x 768 patches/map = 2304 total patches,
     vs. 57 x 768 before).

Everything else (patch/geometry helpers, loss functions, SimpleViT, plotting,
patch-cache builder) is copied over from the original script, adapted only
where the per-realization-file data layout requires it.
"""

import os
import numpy as np
import healpy as hp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import gc
import glob
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##=====================================
##  DATALOADER WORKER COUNT
##=====================================
def _probe_num_workers(max_cap=8):
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    detected = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    candidate = max(0, min(detected, max_cap))

    class _DummyDS(Dataset):
        def __len__(self):
            return 32
        def __getitem__(self, idx):
            return torch.zeros(1)

    while candidate > 0:
        try:
            _loader = DataLoader(_DummyDS(), batch_size=8, num_workers=candidate)
            for _ in _loader:
                pass
            del _loader
            print(f"[perf] NUM_WORKERS probe succeeded with {candidate} workers (detected {detected} CPUs, cap {max_cap})")
            return candidate
        except Exception as e:
            print(f"[perf] NUM_WORKERS probe failed at {candidate} workers ({e}); trying fewer")
            candidate -= 1

    print("[perf] NUM_WORKERS probe: falling back to 0 (single-process loading)")
    return 0

NUM_WORKERS = _probe_num_workers(max_cap=10)
PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else None
print(f"[perf] Using NUM_WORKERS={NUM_WORKERS}, PREFETCH_FACTOR={PREFETCH_FACTOR}")

##=====================================
##  PARAMETERS
##=====================================
RUN_SEED = 0
torch.manual_seed(RUN_SEED)

NSIDE_MAP      = 2048
NSIDE_PATCH    = 256
LMAX           = 3 * NSIDE_MAP - 1
SMOOTH_WINDOW  = 5   # unused by this data path (no smoothing applied to real FLAMINGO maps); kept only because
                     # downstream helper signatures still reference it

BIN_NAMES = ["0", "1", "2"]
n_bins = len(BIN_NAMES)

n_patches_per_map = 12 * (NSIDE_MAP // NSIDE_PATCH) ** 2

DATA_DIR = "/mnt/beegfs/scoulombe"
os.makedirs(DATA_DIR, exist_ok=True)

# --- fine-tune identity ---
SNAPSHOT_TAG = os.environ.get("SNAPSHOT_TAG", "binned_finetune_flamingo_rot_v1")
print(f"Using SNAPSHOT_TAG='{SNAPSHOT_TAG}' for this fine-tune run's checkpoint/history/resume/plot files")

# PATCH_CACHE_TAG deliberately distinct from the original run's "binned" tag
# so this dataset's patch cache / test realization never collide with, or
# copy-forward from, the original curriculum run's cache.
PATCH_CACHE_TAG = os.environ.get("PATCH_CACHE_TAG", "flamingo_rot4")

# --- where the ORIGINAL run's stage-24 weights live (source for init) ---
SOURCE_SNAPSHOT_TAG = os.environ.get("SOURCE_SNAPSHOT_TAG", "binned_finegrained_v2_bigmodel_n64_v1")
SOURCE_STAGE_LABEL  = int(os.environ.get("SOURCE_STAGE_LABEL", 24))  # 1-indexed, matches "_stage{N}_end.pt"
SOURCE_CHECKPOINT_FILE = os.path.join(DATA_DIR, f"model_weights_{SOURCE_SNAPSHOT_TAG}_stage{SOURCE_STAGE_LABEL}_end.pt")

VAL_FRACTION_OF_REALIZATIONS = 0.1

# --- real FLAMINGO map files (from the accompanying rotate-and-save script) ---
# FIELD_NAMES / ROTATIONS / _rot_tag must stay in lockstep with that script.
FIELD_NAMES = ["T", "v0", "v1", "v2", "tau0", "tau1", "tau2", "cmbnoise"]
ROTATIONS = [
    (45, 45, 0),
    (45, 135, 0),
    (135, 45, 0),
    (135, 135, 0),
]

def _rot_tag(rot):
    a, b, c = rot
    return f"rot{a}_{b}_{c}"

# The un-rotated original map, per field -- read-only source, and the
# TEST set for this run (never used in train/val).
FLAMINGO_ORIGINAL_PATHS = {
    name: os.path.join(DATA_DIR, f"flamingo_real_{name}_nside{NSIDE_MAP}.npy")
    for name in FIELD_NAMES
}

# The 4 rotated maps, per field -- this is the entire train/val pool.
# Realization index i (0..3) corresponds to ROTATIONS[i], so
# REAL_FILES[field][i] is a single (npix,) map, NOT a stacked array --
# the patch-cache builder below loads each one individually per realization
# rather than mmap-slicing a stacked file (see build_and_save_stage_chunked_binned).
REAL_FILES = {
    name: [os.path.join(DATA_DIR, f"flamingo_real_{name}_nside{NSIDE_MAP}_{_rot_tag(rot)}.npy") for rot in ROTATIONS]
    for name in FIELD_NAMES
}

N_REALIZATIONS = len(ROTATIONS)  # = 4; fixed by how many rotated maps exist

# TEST_FILES points straight at the real, un-rotated original maps -- no
# generation step, they must already exist (see _validate_flamingo_files).
TEST_FILES = dict(FLAMINGO_ORIGINAL_PATHS)

# Loss weights: reused as-is from the original pipeline (same schema/scale).
LOSS_SCALE_FILE = os.path.join(DATA_DIR, "binned_loss_scale.npz")

def _validate_flamingo_files():
    missing = [p for p in TEST_FILES.values() if not os.path.exists(p)]
    for name in FIELD_NAMES:
        missing += [p for p in REAL_FILES[name] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing FLAMINGO source/rotated map file(s):\n  " + "\n  ".join(missing) +
            "\nRun the rotate-and-save script first to generate the rotated maps, and make "
            "sure the un-rotated flamingo_real_{name}_nside{NSIDE}.npy originals exist."
        )
    print(f"All {len(FIELD_NAMES)} fields x ({len(ROTATIONS)} rotations + 1 original) FLAMINGO files found.")

_validate_flamingo_files()

# --- full EPS_LIST kept only so we can reconstruct the exact stage-24
# mixture below; it is NOT used as a curriculum during fine-tuning.
EPS_LIST = [0.0, 0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.04375,
            0.05, 0.05625, 0.0625, 0.06875, 0.075, 0.08125, 0.0875, 0.09375,
            0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5]
EPS_MAX = max(EPS_LIST)

TIER1_RETIRE = {0.00625, 0.01875, 0.03125, 0.04375, 0.05625, 0.06875, 0.08125, 0.09375}
TIER2_RETIRE = {0.0125, 0.0375, 0.0625, 0.0875}

N_VAL_REALIZATIONS = max(1, int(VAL_FRACTION_OF_REALIZATIONS * N_REALIZATIONS))
VAL_REALIZATIONS   = list(range(N_REALIZATIONS - N_VAL_REALIZATIONS, N_REALIZATIONS))
TRAIN_REALIZATIONS = list(range(0, N_REALIZATIONS - N_VAL_REALIZATIONS))
print(f"Train realizations: {TRAIN_REALIZATIONS}")
print(f"Val realizations:   {VAL_REALIZATIONS}")

##=====================================
##  moving_average
##=====================================
def moving_average(y, window):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    y_padded = np.pad(y, pad, mode="edge")
    smoothed = np.convolve(y_padded, kernel, mode="same")
    return smoothed[pad:pad + len(y)]

##=====================================
##  PATCH / HEALPIX GEOMETRY FUNCTIONS  (unchanged)
##=====================================
def precompute_face_indices(nside):
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix, dtype=np.int32)
    x, y, f = hp.pix2xyf(nside, ipix, nest=True)
    face_idx = {}
    for face_id in range(12):
        mask = f == face_id
        face_idx[face_id] = (ipix[mask].astype(np.int32), x[mask].astype(np.int32), y[mask].astype(np.int32))
    return face_idx

FACE_IDX = precompute_face_indices(NSIDE_MAP)

def get_face_2d(map_nest, nside, face_id, face_idx=None):
    if face_idx is not None:
        ipix_f, x_f, y_f = face_idx[face_id]
        face_img = np.full((nside, nside), np.nan, dtype=map_nest.dtype)
        face_img[y_f, x_f] = map_nest[ipix_f]
        assert not np.isnan(face_img).any(), f"Face {face_id} has unfilled pixels — reshape failed"
        return face_img
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    x, y, f = hp.pix2xyf(nside, ipix, nest=True)
    mask = f == face_id
    face_img = np.full((nside, nside), np.nan)
    face_img[y[mask], x[mask]] = map_nest[ipix[mask]]
    assert not np.isnan(face_img).any(), f"Face {face_id} has unfilled pixels — reshape failed"
    return face_img

def extract_all_patches_from_map(map_nest, nside_map, nside_patch, face_idx=FACE_IDX):
    n_per_side = nside_map // nside_patch
    n_patches = 12 * n_per_side * n_per_side
    out = np.empty((n_patches, nside_patch, nside_patch), dtype=map_nest.dtype)
    idx = 0
    for face_id in range(12):
        face_img = get_face_2d(map_nest, nside_map, face_id, face_idx)
        for pr in range(n_per_side):
            for pc in range(n_per_side):
                out[idx] = face_img[pr*nside_patch:(pr+1)*nside_patch,
                                     pc*nside_patch:(pc+1)*nside_patch]
                idx += 1
    return out

_PATCH_PIXEL_IDX_CACHE = {}

def precompute_patch_pixel_indices(nside_map, nside_patch, face_idx=FACE_IDX):
    n_per_side = nside_map // nside_patch
    n_patches = 12 * n_per_side * n_per_side
    idx_out = np.empty((n_patches, nside_patch, nside_patch), dtype=np.int64)
    patch_i = 0
    for face_id in range(12):
        ipix_f, x_f, y_f = face_idx[face_id]
        face_grid = np.full((nside_map, nside_map), -1, dtype=np.int64)
        face_grid[y_f, x_f] = ipix_f
        assert not (face_grid == -1).any(), f"Face {face_id} has unfilled pixels — index precompute failed"
        for pr in range(n_per_side):
            for pc in range(n_per_side):
                idx_out[patch_i] = face_grid[pr*nside_patch:(pr+1)*nside_patch,
                                              pc*nside_patch:(pc+1)*nside_patch]
                patch_i += 1
    return idx_out

def reassemble_from_stack(patch_stack, nside_map, nside_patch):
    key = (nside_map, nside_patch)
    if key not in _PATCH_PIXEL_IDX_CACHE:
        _PATCH_PIXEL_IDX_CACHE[key] = precompute_patch_pixel_indices(nside_map, nside_patch)
    pix_idx = _PATCH_PIXEL_IDX_CACHE[key]
    assert patch_stack.shape == pix_idx.shape, (
        f"patch_stack shape {patch_stack.shape} does not match expected {pix_idx.shape}"
    )
    out = np.full(hp.nside2npix(nside_map), np.nan, dtype=patch_stack.dtype)
    out[pix_idx.ravel()] = patch_stack.ravel()
    assert not np.isnan(out).any(), "Some pixels were not filled during reassembly!"
    return out

def reassemble_channels_from_stack(patch_stack_multi, nside_map, nside_patch):
    n_channels = patch_stack_multi.shape[1]
    return [reassemble_from_stack(patch_stack_multi[:, c], nside_map, nside_patch)
            for c in range(n_channels)]

def reconstruct_full_maps_multi(model, Tnoise_map_nest, tau_maps_nest, nside_map, nside_patch, eps, batch_size=256):
    Tnoise_stack = extract_all_patches_from_map(Tnoise_map_nest, nside_map, nside_patch)
    tau_stacks = [extract_all_patches_from_map(t, nside_map, nside_patch) for t in tau_maps_nest]

    n_patches = Tnoise_stack.shape[0]
    preds = np.empty((n_patches, n_bins, nside_patch, nside_patch), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_patches, batch_size):
            end = min(start + batch_size, n_patches)
            inp = np.stack([Tnoise_stack[start:end]] + [ts[start:end] for ts in tau_stacks], axis=1)
            inp_t = torch.tensor(inp, dtype=torch.float32).to(device)
            eps_t = torch.full((inp_t.shape[0],), eps, dtype=torch.float32, device=device)
            pred = model(inp_t, eps_t).cpu().numpy()
            preds[start:end] = pred

    return reassemble_channels_from_stack(preds, nside_map, nside_patch)

print("Patch/geometry functions ready")

##=====================================
##  HELD-OUT TEST SET: the real, un-rotated FLAMINGO original map
##
##  Unlike the original script, there's nothing to generate here -- TEST_FILES
##  already points at real source maps on disk (validated above in
##  _validate_flamingo_files()). This function is kept only so the rest of
##  the pipeline's structure/log messages stay parallel to the original
##  script; it does no synthesis.
##=====================================
def generate_test_realization():
    missing = [p for p in TEST_FILES.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Held-out test files (real, un-rotated FLAMINGO maps) missing:\n  " + "\n  ".join(missing)
        )
    print("Held-out test set = real un-rotated FLAMINGO original maps (no synthesis needed):")
    for k, v in TEST_FILES.items():
        print(f"  {k}: {v}")

generate_test_realization()

##=====================================
##  PATCH CACHE (RESUMABLE, SEQUENTIAL, 4 STACKED FILES) FOR THE ROTATED FLAMINGO DATASET
##=====================================
def patch_cache_filename_base_for(n_realizations):
    return os.path.join(DATA_DIR, f"patch_cache_nmap{NSIDE_MAP}_npatch{NSIDE_PATCH}_nreal{n_realizations}_{PATCH_CACHE_TAG}")

def patch_cache_filename_base():
    return patch_cache_filename_base_for(N_REALIZATIONS)

def _cache_files_complete(n_realizations):
    base = patch_cache_filename_base_for(n_realizations)
    suffixes = ["_tau_stack.npy", "_v_target_stack.npy", "_noise.npy", "_T.npy"]
    return all(os.path.exists(base + s) for s in suffixes)

def find_source_cache_n(target_n):
    pattern = os.path.join(
        DATA_DIR, f"patch_cache_nmap{NSIDE_MAP}_npatch{NSIDE_PATCH}_nreal*_{PATCH_CACHE_TAG}_tau_stack.npy"
    )
    found = []
    for path in glob.glob(pattern):
        m = re.search(rf"_nreal(\d+)_{re.escape(PATCH_CACHE_TAG)}_tau_stack\.npy$", path)
        if not m:
            continue
        n_candidate = int(m.group(1))
        if n_candidate < target_n and _cache_files_complete(n_candidate):
            found.append(n_candidate)
    return max(found) if found else None

def _copy_memmap_prefix(src_path, dst_memmap, n_rows, chunk_size=2000):
    src = np.load(src_path, mmap_mode="r")
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        dst_memmap[start:end] = np.array(src[start:end])
    del src

def build_and_save_stage_chunked_binned():
    filename_base = patch_cache_filename_base()
    final_tau_file   = filename_base + "_tau_stack.npy"
    final_v_file     = filename_base + "_v_target_stack.npy"
    final_noise_file = filename_base + "_noise.npy"
    final_T_file     = filename_base + "_T.npy"

    if all(os.path.exists(f) for f in [final_tau_file, final_v_file, final_noise_file, final_T_file]):
        print(f"{filename_base} already exists, skipping generation")
        return filename_base

    # NOTE: REAL_FILES[field] is a list of N_REALIZATIONS individual (npix,)
    # map files (one per rotation), not a single stacked array -- so each
    # realization is np.load()'ed directly by index below rather than
    # mmap-sliced.
    total_patches = N_REALIZATIONS * n_patches_per_map

    tmp_tau_file   = filename_base + "_tmp_tau_stack.npy"
    tmp_v_file     = filename_base + "_tmp_v_target_stack.npy"
    tmp_noise_file = filename_base + "_tmp_noise.npy"
    tmp_T_file     = filename_base + "_tmp_T.npy"
    progress_file  = filename_base + "_progress.txt"

    if os.path.exists(progress_file) and os.path.exists(tmp_tau_file):
        with open(progress_file) as f:
            start_i = int(f.read().strip())
        print(f"Resuming in-progress build for n={N_REALIZATIONS} from realization {start_i}")
        tau_out   = np.lib.format.open_memmap(tmp_tau_file, mode="r+")
        v_out     = np.lib.format.open_memmap(tmp_v_file, mode="r+")
        noise_out = np.lib.format.open_memmap(tmp_noise_file, mode="r+")
        T_out     = np.lib.format.open_memmap(tmp_T_file, mode="r+")
    else:
        tau_out   = np.lib.format.open_memmap(tmp_tau_file, mode="w+", dtype=np.float32,
                                               shape=(total_patches, n_bins, NSIDE_PATCH, NSIDE_PATCH))
        v_out     = np.lib.format.open_memmap(tmp_v_file, mode="w+", dtype=np.float32,
                                               shape=(total_patches, n_bins, NSIDE_PATCH, NSIDE_PATCH))
        noise_out = np.lib.format.open_memmap(tmp_noise_file, mode="w+", dtype=np.float32,
                                               shape=(total_patches, NSIDE_PATCH, NSIDE_PATCH))
        T_out     = np.lib.format.open_memmap(tmp_T_file, mode="w+", dtype=np.float32,
                                               shape=(total_patches, NSIDE_PATCH, NSIDE_PATCH))

        source_n = find_source_cache_n(N_REALIZATIONS)
        if source_n is not None:
            source_base = patch_cache_filename_base_for(source_n)
            source_patches = source_n * n_patches_per_map
            print(f"Found existing complete cache for n={source_n} -- copying "
                  f"{source_patches} patches forward instead of reprocessing them")
            _copy_memmap_prefix(source_base + "_tau_stack.npy", tau_out, source_patches)
            _copy_memmap_prefix(source_base + "_v_target_stack.npy", v_out, source_patches)
            _copy_memmap_prefix(source_base + "_noise.npy", noise_out, source_patches)
            _copy_memmap_prefix(source_base + "_T.npy", T_out, source_patches)
            tau_out.flush(); v_out.flush(); noise_out.flush(); T_out.flush()
            start_i = source_n
            with open(progress_file, "w") as f:
                f.write(str(start_i))
            print(f"Copy-forward complete -- will process realizations {start_i}..{N_REALIZATIONS - 1} fresh")
        else:
            start_i = 0
            print(f"No smaller existing cache found -- building all {N_REALIZATIONS} realizations from scratch")

    for i in range(start_i, N_REALIZATIONS):
        s = slice(i * n_patches_per_map, (i + 1) * n_patches_per_map)

        for bi, b in enumerate(BIN_NAMES):
            v_ring = np.load(REAL_FILES[f"v{b}"][i])
            v_nest = hp.reorder(v_ring, r2n=True)
            v_out[s, bi] = extract_all_patches_from_map(v_nest, NSIDE_MAP, NSIDE_PATCH)
            del v_ring, v_nest

            tau_ring = np.load(REAL_FILES[f"tau{b}"][i])
            tau_nest = hp.reorder(tau_ring, r2n=True)
            tau_out[s, bi] = extract_all_patches_from_map(tau_nest, NSIDE_MAP, NSIDE_PATCH)
            del tau_ring, tau_nest

        noise_ring = np.load(REAL_FILES["cmbnoise"][i])
        noise_nest = hp.reorder(noise_ring, r2n=True)
        noise_out[s] = extract_all_patches_from_map(noise_nest, NSIDE_MAP, NSIDE_PATCH)
        del noise_ring, noise_nest

        T_ring = np.load(REAL_FILES["T"][i])
        T_nest = hp.reorder(T_ring, r2n=True)
        T_out[s] = extract_all_patches_from_map(T_nest, NSIDE_MAP, NSIDE_PATCH)
        del T_ring, T_nest

        tau_out.flush(); v_out.flush(); noise_out.flush(); T_out.flush()
        with open(progress_file, "w") as f:
            f.write(str(i + 1))

        gc.collect()
        print(f"Patched {i+1}/{N_REALIZATIONS} realizations — saved to disk")

    del tau_out, v_out, noise_out, T_out
    gc.collect()

    import shutil
    shutil.move(tmp_tau_file, final_tau_file)
    shutil.move(tmp_v_file, final_v_file)
    shutil.move(tmp_noise_file, final_noise_file)
    shutil.move(tmp_T_file, final_T_file)
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"Saved final files: {final_tau_file}, {final_v_file}, {final_noise_file}, {final_T_file}")
    return filename_base

def compute_stats_streaming(filepath, chunk_size=1000):
    arr = np.load(filepath, mmap_mode="r")
    n = len(arr)
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for i in range(0, n, chunk_size):
        chunk = np.array(arr[i:i+chunk_size], dtype=np.float64)
        total_sum += chunk.sum()
        total_sq_sum += (chunk**2).sum()
        total_count += chunk.size
    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean**2
    std = np.sqrt(var)
    del arr
    return float(mean), float(std)

filename_base = build_and_save_stage_chunked_binned()
tau_stack_file = filename_base + "_tau_stack.npy"
v_target_stack_file = filename_base + "_v_target_stack.npy"
noise_file = filename_base + "_noise.npy"
T_file = filename_base + "_T.npy"

for label, path in [("tau_stack", tau_stack_file), ("v_target_stack", v_target_stack_file),
                     ("noise", noise_file), ("T", T_file)]:
    m, s = compute_stats_streaming(path)
    print(f"{label}: mean={m:.4f} std={s:.4f}")

##=====================================
##  LOAD LOSS WEIGHTS
##=====================================
if not os.path.exists(LOSS_SCALE_FILE):
    raise FileNotFoundError(f"{LOSS_SCALE_FILE} not found — run the loss-weight notebook first.")

_scales = np.load(LOSS_SCALE_FILE)
weight_pixel    = float(_scales["weight_pixel"])
weight_patch    = float(_scales["weight_patch"])
weight_spectral = float(_scales["weight_spectral"])
N_SPEC_BINS   = int(_scales["n_spec_bins"])
SPEC_LOSS_EPS = float(_scales["spec_loss_eps"])
print(f"Loaded loss weights: weight_pixel={weight_pixel:.6e}, weight_patch={weight_patch:.6e}, "
      f"weight_spectral={weight_spectral:.6e}  (n_spec_bins={N_SPEC_BINS}, spec_loss_eps={SPEC_LOSS_EPS})")

##=====================================
##  DATASET CLASSES  (unchanged)
##=====================================
class VTauNoiseDatasetFromDisk(Dataset):
    def __init__(self, tau_stack_file, v_target_stack_file, noise_file, T_file):
        self.tau_stack = np.load(tau_stack_file, mmap_mode="r")
        self.v_target_stack = np.load(v_target_stack_file, mmap_mode="r")
        self.noise = np.load(noise_file, mmap_mode="r")
        self.T = np.load(T_file, mmap_mode="r")

    def __len__(self):
        return len(self.T)

    def get_raw(self, idx):
        return (np.array(self.tau_stack[idx]), np.array(self.v_target_stack[idx]),
                np.array(self.noise[idx]), np.array(self.T[idx]))

class MixedEpsilonDataset(Dataset):
    """Same mixing mechanism as the original script, but in this fine-tune
    script it's always driven by a single, fixed weight dict (set once via
    set_weights() and never changed for the rest of the run)."""
    def __init__(self, base_dataset, index_pool, eps_list, samples_per_epoch: int, seed=RUN_SEED):
        self.base_dataset = base_dataset
        self.index_pool = np.asarray(index_pool)
        self.eps_keys = list(eps_list)
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self._plan = None

    def set_weights(self, weights: dict):
        w = np.array([weights.get(k, 0.0) for k in self.eps_keys], dtype=np.float64)
        assert np.isclose(w.sum(), 1.0), f"weights sum to {w.sum()}, expected 1.0"
        self.weights = w

    def resample(self, samples_per_epoch=None):
        assert self.weights is not None, "call set_weights() before resample()"
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        chosen_eps = self.rng.choice(self.eps_keys, size=self.samples_per_epoch, p=self.weights)
        chosen_idx = self.rng.choice(self.index_pool, size=self.samples_per_epoch, replace=True)
        self._plan = list(zip(chosen_eps, chosen_idx))

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        eps, local_idx = self._plan[idx]
        tau3, v3, noise, T = self.base_dataset.get_raw(local_idx)
        Tnoise = (T + eps * noise).astype(np.float32)
        inp = torch.from_numpy(
            np.concatenate([Tnoise[None, :, :], tau3.astype(np.float32)], axis=0)
        )
        target = torch.from_numpy(v3.astype(np.float32))
        return inp, target, torch.tensor(eps, dtype=torch.float32)

TRAIN_IDX = np.concatenate([np.arange(r * n_patches_per_map, (r + 1) * n_patches_per_map) for r in TRAIN_REALIZATIONS])
VAL_IDX   = np.concatenate([np.arange(r * n_patches_per_map, (r + 1) * n_patches_per_map) for r in VAL_REALIZATIONS])

full_dataset = VTauNoiseDatasetFromDisk(tau_stack_file, v_target_stack_file, noise_file, T_file)
print(f"Built single eps-agnostic patch dataset for the new dataset, shared across eps values: {EPS_LIST}")

##=====================================
##  LOSS FUNCTIONS  (unchanged)
##=====================================
def pixel_loss_fn(pred, target):
    return torch.mean((pred - target) ** 2)

def patch_loss_fn(pred, target):
    pred_mean   = pred.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))
    return torch.mean((pred_mean - target_mean) ** 2)

_RADIAL_BIN_CACHE = {}

def get_radial_bins(patch_size, n_bins, torch_device):
    key = (patch_size, n_bins, str(torch_device))
    if key in _RADIAL_BIN_CACHE:
        return _RADIAL_BIN_CACHE[key]
    freqs_y = torch.fft.fftfreq(patch_size)
    freqs_x = torch.fft.rfftfreq(patch_size)
    ky, kx = torch.meshgrid(freqs_y, freqs_x, indexing='ij')
    k = torch.sqrt(kx**2 + ky**2)
    k_flat = k.flatten()
    nonzero = k_flat[k_flat > 0]
    kmin = nonzero.min()
    kmax = k_flat.max()
    bin_edges = torch.logspace(torch.log10(kmin), torch.log10(kmax), n_bins + 1)
    bin_idx = torch.bucketize(k_flat, bin_edges) - 1
    bin_idx = bin_idx.clamp(0, n_bins - 1).long().to(torch_device)
    _RADIAL_BIN_CACHE[key] = bin_idx
    return bin_idx

def radial_power_spectrum(patch_batch, n_bins=N_SPEC_BINS):
    N, H, W = patch_batch.shape
    fft = torch.fft.rfft2(patch_batch)
    power = fft.real**2 + fft.imag**2
    bin_idx = get_radial_bins(H, n_bins, patch_batch.device)
    power_flat = power.reshape(N, -1)
    binned = torch.zeros(N, n_bins, device=patch_batch.device, dtype=power.dtype)
    binned.index_add_(1, bin_idx, power_flat)
    counts = torch.zeros(n_bins, device=patch_batch.device, dtype=power.dtype)
    counts.index_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=power.dtype))
    counts = counts.clamp(min=1.0)
    binned = binned / counts.unsqueeze(0)
    return binned

def spectral_loss_fn(pred, target, n_bins=N_SPEC_BINS, eps=SPEC_LOSS_EPS):
    B, C, H, W = pred.shape
    pred_flat   = pred.reshape(B * C, H, W)
    target_flat = target.reshape(B * C, H, W)

    P_pred = radial_power_spectrum(pred_flat, n_bins=n_bins)
    P_true = radial_power_spectrum(target_flat, n_bins=n_bins)

    log_pred = torch.log(P_pred + eps)
    log_true = torch.log(P_true + eps)

    return torch.mean((log_pred - log_true) ** 2)

def combined_loss_fn(pred, target):
    l_pixel = pixel_loss_fn(pred, target)
    l_patch = patch_loss_fn(pred, target)
    l_spectral = spectral_loss_fn(pred, target)
    total = weight_pixel * l_pixel + weight_patch * l_patch + weight_spectral * l_spectral
    return total, l_pixel.detach(), l_patch.detach(), l_spectral.detach()

print("Component losses ready: pixel MSE, patch-average MSE, spectral (log-space radial power) MSE — all channel-averaged")

##=====================================
##  ViT MODEL  (unchanged architecture)
##=====================================
class SimpleViT(nn.Module):
    def __init__(self, img_size, patch_size=8, in_ch=4, out_ch=3,
                 embed_dim=64, depth=4, n_heads=4, mlp_ratio=4, eps_max=EPS_MAX):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.n_tokens = self.grid * self.grid
        self.out_ch = out_ch
        self.eps_max = eps_max
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, embed_dim) * 0.02)
        self.eps_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * mlp_ratio,
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_ch * patch_size * patch_size)

    def forward(self, x, eps):
        B, C, H, W = x.shape
        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        eps_norm = (eps.float() / self.eps_max).view(B, 1)
        eps_emb = self.eps_embed(eps_norm)
        tokens = tokens + eps_emb.unsqueeze(1)

        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        out = self.head(tokens)
        p = self.patch_size
        out = out.reshape(B, self.grid, self.grid, self.out_ch, p, p)
        out = out.permute(0, 3, 1, 4, 2, 5)
        out = out.reshape(B, self.out_ch, self.grid * p, self.grid * p)
        return out

##=====================================
##  FIXED EPS MIXTURE (no curriculum)
##
##  We reconstruct the exact weight distribution that was active at the end
##  of stage 24 in the original curriculum run, and use it, UNCHANGED, for
##  the entire fine-tuning run. This keeps the eps distribution the model
##  sees continuous across the "switch" from stage-24 curriculum training to
##  fixed-mixture fine-tuning. Edit FIXED_EPS_WEIGHTS directly if you'd
##  rather use a different fixed mixture.
##=====================================
def compute_stage_weights(active, tier1_set, tier2_set, prev_weights, eps0_weight,
                           base_weight=0.12, retire_floor=0.02):
    remaining_budget = 1.0 - eps0_weight
    durable = [e for e in active if e not in tier1_set and e not in tier2_set]
    tier1   = [e for e in active if e in tier1_set]
    tier2   = [e for e in active if e in tier2_set]

    weights = {e: prev_weights.get(e, base_weight) for e in active}
    deficit = sum(weights.values()) - remaining_budget

    if deficit > 1e-9:
        for e in tier1:
            if deficit <= 1e-9:
                break
            take = min(weights[e] - retire_floor, deficit)
            if take > 0:
                weights[e] -= take
                deficit -= take
        tier1_floored = all(weights[e] <= retire_floor + 1e-9 for e in tier1) if tier1 else True
        if tier1_floored:
            for e in tier2:
                if deficit <= 1e-9:
                    break
                take = min(weights[e] - retire_floor, deficit)
                if take > 0:
                    weights[e] -= take
                    deficit -= take
        if deficit > 1e-9 and durable:
            per = deficit / len(durable)
            for e in durable:
                weights[e] = max(0.0, weights[e] - per)
    elif deficit < -1e-9:
        surplus = -deficit
        pool = durable if durable else (tier1 if tier1 else tier2)
        if pool:
            per = surplus / len(pool)
            for e in pool:
                weights[e] += per

    return weights

def build_curriculum_stages(eps_list, samples_per_eps, eps0_start=1.0, eps0_end=0.3,
                             base_weight=0.12, retire_floor=0.02):
    nonzero_eps = eps_list[1:]
    n_stages = len(eps_list)

    active = []
    prev_weights = {}
    stages = [{"weights": {0.0: 1.0}, "samples_per_epoch": samples_per_eps}]

    for stage_i in range(2, n_stages + 1):
        new_eps = nonzero_eps[stage_i - 2]
        active.append(new_eps)

        eps0_weight = eps0_start - (eps0_start - eps0_end) * (stage_i - 1) / (n_stages - 1)
        nonzero_weights = compute_stage_weights(active, TIER1_RETIRE, TIER2_RETIRE, prev_weights,
                                                  eps0_weight, base_weight=base_weight, retire_floor=retire_floor)

        weights = {0.0: eps0_weight, **nonzero_weights}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        stages.append({"weights": weights, "samples_per_epoch": samples_per_eps * len(weights)})
        prev_weights = nonzero_weights

    return stages

# One full pass over the currently available train patches per eps value.
SAMPLES_PER_EPS = len(TRAIN_REALIZATIONS) * n_patches_per_map
print(f"SAMPLES_PER_EPS={SAMPLES_PER_EPS} (derived from {len(TRAIN_REALIZATIONS)} train "
      f"realizations x {n_patches_per_map} patches/map)")

# Rebuild the ORIGINAL curriculum purely to extract the stage-24 weight
# dict -- we never iterate through these stages during fine-tuning.
_ORIGINAL_CURRICULUM_STAGES = build_curriculum_stages(EPS_LIST, samples_per_eps=SAMPLES_PER_EPS)
_source_stage_idx0 = SOURCE_STAGE_LABEL - 1  # 0-indexed
assert 0 <= _source_stage_idx0 < len(_ORIGINAL_CURRICULUM_STAGES), (
    f"SOURCE_STAGE_LABEL={SOURCE_STAGE_LABEL} out of range for a {len(_ORIGINAL_CURRICULUM_STAGES)}-stage curriculum"
)
FIXED_EPS_WEIGHTS = dict(_ORIGINAL_CURRICULUM_STAGES[_source_stage_idx0]["weights"])
_w_sum = sum(FIXED_EPS_WEIGHTS.values())
assert np.isclose(_w_sum, 1.0), f"FIXED_EPS_WEIGHTS sum to {_w_sum}, expected 1.0"
print(f"Fixed (non-curriculum) eps mixture for this fine-tune, matching end of original stage {SOURCE_STAGE_LABEL}:")
print(f"  {{ {', '.join(f'{k}: {v:.4f}' for k, v in sorted(FIXED_EPS_WEIGHTS.items()))} }}")

FIXED_SAMPLES_PER_EPOCH = SAMPLES_PER_EPS * len(FIXED_EPS_WEIGHTS)

# --- easily-editable hyperparameters ---
PATIENCE     = 10
MIN_DELTA    = 0.0005
VAL_FRACTION = 0.15
BATCH_SIZE   = 64   # lowered from 256: train pool is only ~3 realizations x 768
                     # patches/map = 2304 total patches, vs. 57x768 in the
                     # original curriculum run
MAX_EPOCHS   = 1000
GRAD_CLIP_MAX_NORM = 1.0
LEARNING_RATE = 1e-4  # lower than the original 1e-3 -- fine-tuning from a
                       # curriculum-trained checkpoint, not training from
                       # scratch. Raise back to 1e-3 if you want a more
                       # aggressive adaptation.
SAFETY_CHECKPOINT_EVERY = 5

##=====================================
##  MOLLWEIDE / PATCH SNAPSHOT PLOTTING  (unchanged)
##=====================================
def plot_mollweide_multi(v_true_ring, recon_rings: dict, label=None, filename_tag=None):
    eps_list = sorted(recon_rings.keys())
    n_cols = 1 + len(eps_list)
    fig = plt.figure(figsize=(5 * n_cols, 10))
    vmax = np.nanpercentile(np.abs(v_true_ring), 99)

    hp.mollview(v_true_ring, fig=fig, sub=(2, n_cols, 1),
                title="True v", min=-vmax, max=vmax, cmap="RdBu_r")

    for i, eps in enumerate(eps_list):
        recon_ring = recon_rings[eps]
        col = i + 2
        hp.mollview(recon_ring, fig=fig, sub=(2, n_cols, col),
                    title=f"Reconstructed (eps={eps})", min=-vmax, max=vmax, cmap="RdBu_r")
        residual = v_true_ring - recon_ring
        res_vmax = np.nanpercentile(np.abs(residual), 99)
        hp.mollview(residual, fig=fig, sub=(2, n_cols, n_cols + col),
                    title=f"Residual (eps={eps})", min=-res_vmax, max=res_vmax, cmap="RdBu_r")

    suptitle = "Mollweide comparison across epsilons" + (f" — {label}" if label else "")
    fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    tag_str = f"_{filename_tag}" if filename_tag else ""
    fname = f"mollweide_multi_epoch{label.replace(' ', '_').replace(',', '') if label else 'unlabeled'}{tag_str}.png"
    fig_path = os.path.join(DATA_DIR, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-epsilon Mollweide comparison to {fig_path}")

PATCH_COMPARISON_LOCATIONS = [
    {"face": 0,  "row": 0, "col": 0},
    {"face": 5,  "row": 4, "col": 4},
]

def extract_all_patches(map_nest, nside_map, nside_patch, face_idx=FACE_IDX):
    all_patches = []
    for face_id in range(12):
        face_img = get_face_2d(map_nest, nside_map, face_id, face_idx)
        n_per_side = nside_map // nside_patch
        for pr in range(n_per_side):
            for pc in range(n_per_side):
                patch = face_img[pr*nside_patch:(pr+1)*nside_patch, pc*nside_patch:(pc+1)*nside_patch]
                all_patches.append({"face": face_id, "row": pr, "col": pc, "data": patch})
    return all_patches

def plot_patch_multi(v_true_nest, recon_nests: dict, label=None, filename_tag=None,
                      locations=PATCH_COMPARISON_LOCATIONS):
    eps_list = sorted(recon_nests.keys())
    n_cols = 1 + len(eps_list)
    n_rows = len(locations)

    true_patches = extract_all_patches(v_true_nest, NSIDE_MAP, NSIDE_PATCH, FACE_IDX)
    true_lookup = {(p["face"], p["row"], p["col"]): p["data"] for p in true_patches}

    recon_lookups = {}
    for eps, recon_nest in recon_nests.items():
        recon_patches = extract_all_patches(recon_nest, NSIDE_MAP, NSIDE_PATCH, FACE_IDX)
        recon_lookups[eps] = {(p["face"], p["row"], p["col"]): p["data"] for p in recon_patches}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for row, loc in enumerate(locations):
        key = (loc["face"], loc["row"], loc["col"])
        true_patch = true_lookup[key]
        vmax = np.nanpercentile(np.abs(true_patch), 99)

        axes[row, 0].imshow(true_patch, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        axes[row, 0].set_title(f"true (face={key[0]},row={key[1]},col={key[2]})", fontsize=9)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        for col, eps in enumerate(eps_list, start=1):
            pred_patch = recon_lookups[eps][key]
            mse = float(np.mean((pred_patch - true_patch) ** 2))
            axes[row, col].imshow(pred_patch, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            axes[row, col].set_title(f"eps={eps} (MSE={mse:.3e})", fontsize=9)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    suptitle = "Patch comparison across epsilons" + (f" — {label}" if label else "")
    fig.suptitle(suptitle, y=1.0)
    plt.tight_layout()
    tag_str = f"_{filename_tag}" if filename_tag else ""
    fname = f"patchcompare_multi_epoch{label.replace(' ', '_').replace(',', '') if label else 'unlabeled'}{tag_str}.png"
    fig_path = os.path.join(DATA_DIR, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-epsilon patch comparison to {fig_path}")

def test_snapshot(model, eps_to_test: list, label=None, filename_tag=None):
    v_test_rings   = {b: np.load(TEST_FILES[f"v{b}"]) for b in BIN_NAMES}
    tau_test_rings = {b: np.load(TEST_FILES[f"tau{b}"]) for b in BIN_NAMES}
    noise_test_ring = np.load(TEST_FILES["cmbnoise"])
    T_test_ring = np.load(TEST_FILES["T"])

    tau_test_nests = {b: hp.reorder(tau_test_rings[b], r2n=True) for b in BIN_NAMES}
    v_test_nests   = {b: hp.reorder(v_test_rings[b], r2n=True) for b in BIN_NAMES}

    Cl_true = {b: hp.anafast(v_test_rings[b], lmax=LMAX) for b in BIN_NAMES}

    model.eval()
    Cl_preds = {b: {} for b in BIN_NAMES}
    recon_rings = {b: {} for b in BIN_NAMES}
    recon_nests = {b: {} for b in BIN_NAMES}

    for eps in eps_to_test:
        Tnoise_test_ring = (T_test_ring + eps * noise_test_ring).astype(np.float32)
        Tnoise_test_nest = hp.reorder(Tnoise_test_ring, r2n=True)

        tau_maps_nest = [tau_test_nests[b] for b in BIN_NAMES]
        recon_nests_list = reconstruct_full_maps_multi(
            model, Tnoise_test_nest, tau_maps_nest, NSIDE_MAP, NSIDE_PATCH, eps=eps
        )

        for bi, b in enumerate(BIN_NAMES):
            recon_nest = recon_nests_list[bi]
            recon_ring = hp.reorder(recon_nest, n2r=True)
            Cl_preds[b][eps] = hp.anafast(recon_ring, lmax=LMAX)
            recon_rings[b][eps] = recon_ring
            recon_nests[b][eps] = recon_nest

        del Tnoise_test_ring, Tnoise_test_nest
        gc.collect()

    for b in BIN_NAMES:
        bin_tag = f"{filename_tag}_bin{b}" if filename_tag else f"bin{b}"
        plot_mollweide_multi(v_test_rings[b], recon_rings[b], label=label, filename_tag=bin_tag)
        plot_patch_multi(v_test_nests[b], recon_nests[b], label=label, filename_tag=bin_tag)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(Cl_true[b], color='black', linewidth=2, label="true")
        for eps in eps_to_test:
            ax.plot(Cl_preds[b][eps], label=f"eps={eps}", alpha=0.8)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylim(1e-7, None)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$C_\ell$")
        ax.legend()
        title = f"Reconstruction quality — bin {b}" + (f" — {label}" if label else "")
        ax.set_title(title)
        plt.tight_layout()
        tag_str = f"_{bin_tag}"
        fig_path = os.path.join(DATA_DIR, f"snapshot_epoch{label.replace(' ', '_').replace(',', '') if label else 'unlabeled'}{tag_str}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved snapshot plot to {fig_path}")

    del v_test_rings, tau_test_rings, noise_test_ring, T_test_ring
    del v_test_nests, tau_test_nests, recon_rings, recon_nests
    gc.collect()

    return Cl_true, Cl_preds

##=====================================
##  RESUME STATE  (simplified: no stage tracking needed, single fixed phase)
##=====================================
RESUME_STATE_FILE = os.path.join(DATA_DIR, f"resume_state_{SNAPSHOT_TAG}.npz")

def save_resume_state(next_epoch, epochs_since_improvement, best_val_loss,
                       global_best_val_loss, global_best_epoch):
    np.savez(RESUME_STATE_FILE,
             next_epoch=next_epoch,
             epochs_since_improvement=epochs_since_improvement,
             best_val_loss=best_val_loss,
             global_best_val_loss=global_best_val_loss,
             global_best_epoch=global_best_epoch if global_best_epoch is not None else -1)

def load_resume_state():
    if not os.path.exists(RESUME_STATE_FILE):
        return None
    d = np.load(RESUME_STATE_FILE)
    gbe = int(d["global_best_epoch"])
    return {
        "next_epoch": int(d["next_epoch"]),
        "epochs_since_improvement": int(d["epochs_since_improvement"]),
        "best_val_loss": float(d["best_val_loss"]),
        "global_best_val_loss": float(d["global_best_val_loss"]),
        "global_best_epoch": None if gbe == -1 else gbe,
    }

##=====================================
##  TRAINING LOOP  (single fixed-mixture phase, early stopping via patience)
##=====================================
CHECKPOINT_FILE      = os.path.join(DATA_DIR, f"model_weights_{SNAPSHOT_TAG}.pt")
BEST_CHECKPOINT_FILE = os.path.join(DATA_DIR, f"model_weights_{SNAPSHOT_TAG}_best.pt")
TRAIN_HISTORY_FILE   = os.path.join(DATA_DIR, f"train_history_{SNAPSHOT_TAG}.npz")

FORCE_RESTART = os.environ.get("FORCE_RESTART", "0") == "1"
if FORCE_RESTART:
    print("FORCE_RESTART is set -- ignoring any existing fine-tune checkpoint/history/resume-state "
          "files and starting fresh from epoch 1 (still initialized from the stage-24 source checkpoint).")

model = SimpleViT(img_size=NSIDE_PATCH, in_ch=4, out_ch=3, eps_max=EPS_MAX,
                   embed_dim=128, depth=5).to(device)

if not FORCE_RESTART and os.path.exists(CHECKPOINT_FILE):
    # We've already fine-tuned for at least one epoch under this
    # SNAPSHOT_TAG -- resume from OUR OWN latest checkpoint, not the
    # original stage-24 source checkpoint.
    model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=device))
    model = model.to(device)
    print(f"Resumed fine-tune model weights from {CHECKPOINT_FILE}")
else:
    # First time this fine-tune run has started (or a forced restart):
    # initialize from the original run's stage-24 weights.
    if not os.path.exists(SOURCE_CHECKPOINT_FILE):
        raise FileNotFoundError(
            f"Source checkpoint {SOURCE_CHECKPOINT_FILE} not found. Check SOURCE_SNAPSHOT_TAG "
            f"(='{SOURCE_SNAPSHOT_TAG}') and SOURCE_STAGE_LABEL (={SOURCE_STAGE_LABEL}) against "
            f"what the original run actually produced."
        )
    model.load_state_dict(torch.load(SOURCE_CHECKPOINT_FILE, map_location=device))
    model = model.to(device)
    print(f"Initialized fine-tune model weights from source checkpoint {SOURCE_CHECKPOINT_FILE}")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Using COMBINED patch + spectral loss (channel-averaged, normalized weighted sum), "
      f"FIXED eps mixture (no curriculum), batch_size={BATCH_SIZE}, grad_clip_max_norm={GRAD_CLIP_MAX_NORM}, "
      f"lr={LEARNING_RATE}")

_active_train_subsets = (full_dataset, TRAIN_IDX)
_active_val_subsets   = (full_dataset, VAL_IDX)

train_losses, val_losses = [], []
train_losses_pixel, train_losses_patch, train_losses_spectral = [], [], []
val_losses_pixel, val_losses_patch, val_losses_spectral = [], [], []

if not FORCE_RESTART and os.path.exists(TRAIN_HISTORY_FILE):
    _prev_hist = np.load(TRAIN_HISTORY_FILE)
    train_losses           = list(_prev_hist["train_losses"])
    val_losses              = list(_prev_hist["val_losses"])
    train_losses_pixel      = list(_prev_hist["train_losses_pixel"])
    train_losses_patch      = list(_prev_hist["train_losses_patch"])
    train_losses_spectral   = list(_prev_hist["train_losses_spectral"])
    val_losses_pixel         = list(_prev_hist["val_losses_pixel"])
    val_losses_patch         = list(_prev_hist["val_losses_patch"])
    val_losses_spectral      = list(_prev_hist["val_losses_spectral"])
    print(f"Loaded {len(train_losses)} epochs of prior fine-tune loss history from {TRAIN_HISTORY_FILE} -- will extend, not overwrite")

resume_state = None if FORCE_RESTART else load_resume_state()

train_mixed_ds = MixedEpsilonDataset(*_active_train_subsets, EPS_LIST, samples_per_epoch=FIXED_SAMPLES_PER_EPOCH, seed=RUN_SEED)
train_mixed_ds.set_weights(FIXED_EPS_WEIGHTS)

val_samples = max(1, int(FIXED_SAMPLES_PER_EPOCH * VAL_FRACTION))
val_mixed_ds = MixedEpsilonDataset(*_active_val_subsets, EPS_LIST, samples_per_epoch=val_samples, seed=RUN_SEED + 1000)
val_mixed_ds.set_weights(FIXED_EPS_WEIGHTS)
val_mixed_ds.resample()

if resume_state is not None:
    start_epoch               = resume_state["next_epoch"]
    epochs_since_improvement   = resume_state["epochs_since_improvement"]
    best_val_loss              = resume_state["best_val_loss"]
    global_best_val_loss       = resume_state["global_best_val_loss"]
    global_best_epoch          = resume_state["global_best_epoch"]
    print(f"Resuming fine-tune: next epoch={start_epoch+1}, epochs_since_improvement={epochs_since_improvement}, "
          f"best_val_loss={best_val_loss}, global_best_val_loss={global_best_val_loss}, global_best_epoch={global_best_epoch}")
else:
    start_epoch               = 0
    epochs_since_improvement   = 0
    best_val_loss              = float("inf")
    global_best_val_loss       = float("inf")
    global_best_epoch          = None
    print("No fine-tune resume state found — starting fresh from epoch 1")

def save_all_checkpoints():
    torch.save(model.state_dict(), CHECKPOINT_FILE)
    np.savez(TRAIN_HISTORY_FILE,
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses),
             train_losses_pixel=np.array(train_losses_pixel),
             train_losses_patch=np.array(train_losses_patch),
             train_losses_spectral=np.array(train_losses_spectral),
             val_losses_pixel=np.array(val_losses_pixel),
             val_losses_patch=np.array(val_losses_patch),
             val_losses_spectral=np.array(val_losses_spectral),
             weight_pixel=weight_pixel,
             weight_patch=weight_patch,
             weight_spectral=weight_spectral,
             best_val_loss=global_best_val_loss,
             best_epoch=global_best_epoch)

SNAPSHOT_EVERY_N_EPOCHS = int(os.environ.get("SNAPSHOT_EVERY_N_EPOCHS", 20))

for epoch in range(start_epoch, MAX_EPOCHS):
    train_mixed_ds.resample(FIXED_SAMPLES_PER_EPOCH)
    train_loader = DataLoader(
        train_mixed_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
        num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        val_mixed_ds, batch_size=BATCH_SIZE, pin_memory=True,
        num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR,
    )

    n_train = len(train_mixed_ds)
    n_val   = len(val_mixed_ds)

    model.train()
    epoch_loss = epoch_loss_pixel = epoch_loss_patch = epoch_loss_spectral = 0.0
    for inp, target, eps_batch in train_loader:
        inp, target, eps_batch = inp.to(device), target.to(device), eps_batch.to(device)
        optimizer.zero_grad()
        pred = model(inp, eps_batch)
        loss, l_pixel, l_patch, l_spectral = combined_loss_fn(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()
        epoch_loss += loss.item() * inp.size(0)
        epoch_loss_pixel += l_pixel.item() * inp.size(0)
        epoch_loss_patch += l_patch.item() * inp.size(0)
        epoch_loss_spectral += l_spectral.item() * inp.size(0)
    epoch_loss /= n_train
    epoch_loss_pixel /= n_train
    epoch_loss_patch /= n_train
    epoch_loss_spectral /= n_train
    train_losses.append(epoch_loss)
    train_losses_pixel.append(epoch_loss_pixel)
    train_losses_patch.append(epoch_loss_patch)
    train_losses_spectral.append(epoch_loss_spectral)

    model.eval()
    val_loss = val_loss_pixel = val_loss_patch = val_loss_spectral = 0.0
    with torch.no_grad():
        for inp, target, eps_batch in val_loader:
            inp, target, eps_batch = inp.to(device), target.to(device), eps_batch.to(device)
            pred = model(inp, eps_batch)
            loss, l_pixel, l_patch, l_spectral = combined_loss_fn(pred, target)
            val_loss += loss.item() * inp.size(0)
            val_loss_pixel += l_pixel.item() * inp.size(0)
            val_loss_patch += l_patch.item() * inp.size(0)
            val_loss_spectral += l_spectral.item() * inp.size(0)
    val_loss /= n_val
    val_loss_pixel /= n_val
    val_loss_patch /= n_val
    val_loss_spectral /= n_val
    val_losses.append(val_loss)
    val_losses_pixel.append(val_loss_pixel)
    val_losses_patch.append(val_loss_patch)
    val_losses_spectral.append(val_loss_spectral)

    print(f"[fine-tune] Epoch {epoch+1}  train_total={epoch_loss:.5f}  val_total={val_loss:.5f}  |  "
          f"train[pixel={epoch_loss_pixel:.5f} patch={epoch_loss_patch:.5f} spectral={epoch_loss_spectral:.5f}]  "
          f"val[pixel={val_loss_pixel:.5f} patch={val_loss_patch:.5f} spectral={val_loss_spectral:.5f}]")

    if val_loss < global_best_val_loss:
        global_best_val_loss = val_loss
        global_best_epoch = epoch + 1
        torch.save(model.state_dict(), BEST_CHECKPOINT_FILE)
        print(f"  (new best combined val_loss={val_loss:.5f} at epoch {epoch+1} — saved to {BEST_CHECKPOINT_FILE})")

    if best_val_loss == float("inf"):
        relative_improvement = 1.0
    else:
        relative_improvement = (best_val_loss - val_loss) / best_val_loss

    if relative_improvement > MIN_DELTA:
        best_val_loss = val_loss
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    save_resume_state(
        next_epoch=epoch + 1,
        epochs_since_improvement=epochs_since_improvement,
        best_val_loss=best_val_loss,
        global_best_val_loss=global_best_val_loss,
        global_best_epoch=global_best_epoch,
    )
    save_all_checkpoints()

    if (epoch + 1) % SAFETY_CHECKPOINT_EVERY == 0:
        print(f"  (checkpoint + resume state saved at epoch {epoch+1})")

    if (epoch + 1) % SNAPSHOT_EVERY_N_EPOCHS == 0:
        print(f"\n--- Fine-tune progress snapshot at epoch {epoch+1} ---")
        test_snapshot(
            model,
            eps_to_test=sorted(FIXED_EPS_WEIGHTS.keys()),
            label=f"epoch {epoch+1}",
            filename_tag=SNAPSHOT_TAG
        )

    if epochs_since_improvement >= PATIENCE:
        print(f"\n=== FINE-TUNING COMPLETE: plateaued at epoch {epoch+1} (patience={PATIENCE}) — stopping automatically ===")
        break

print(f"Saved {CHECKPOINT_FILE} and {TRAIN_HISTORY_FILE}")
print(f"Best fine-tune checkpoint: epoch {global_best_epoch}, val_loss={global_best_val_loss:.5f} -> {BEST_CHECKPOINT_FILE}")

del train_loader, val_loader
gc.collect()

print("\n--- Final fine-tune snapshot ---")
test_snapshot(model, eps_to_test=sorted(FIXED_EPS_WEIGHTS.keys()),
              label="final fine-tune sweep", filename_tag=SNAPSHOT_TAG)