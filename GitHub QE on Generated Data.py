"""
Standalone Quadratic Estimator (QE) evaluation for the BINNED (3-tracer)
problem. No ViT is loaded or run anywhere in this script.

This is a variant of the original full-range script. The ONLY substantive
changes from the original are:
  1. QE_LMIN_FILTER / QE_LMAX_FILTER are set to [10, 5000] (inclusive)
     instead of [1, LMAX]. This filter is applied to BOTH inputs of the
     estimator (T_tilde and each bin's tau_tilde) via almxfl, BEFORE the
     alm2map that produces the real-space maps that get multiplied together.
     It also bounds the wigner-3j sum inside compute_Nl. So the QE itself --
     not just the MSE reported afterward -- only ever uses/produces
     information in l=[10, 5000]. Everything outside that band is exactly
     zero in T_tilde_map and Tao_tilde_map.
  2. valid_mask (used for MSE) is changed from ">= QE_LMIN_FILTER" (which
     used to mean "l=1..LMAX") to an explicit two-sided mask matching the
     new filter band exactly, so MSE is only ever computed over the range
     the QE was actually allowed to use.
  3. Output paths (RESULTS_FILE, SUMMARY_PLOT) are moved into a new
     final_qe_generated/ subdirectory with filenames that encode the filter
     range, so this run does NOT overwrite any previous full-range results.
     ENSEMBLE_SPECTRA_CACHE is left unchanged/shared on purpose: the
     ensemble spectra (Cl_tau_avg, Cl_T_signal_avg, Cl_noise_avg) are
     computed from the full unfiltered training realizations and do not
     depend on QE_LMIN_FILTER/QE_LMAX_FILTER at all -- the filter is only
     applied later, at reconstruction time -- so reusing the cache is
     correct and saves recomputing 32x3 anafast calls.
  4. The reconstructed map v_hat is now saved INSIDE RESULTS_FILE (one file
     per run containing cl_true, cl_qe, cl_cross, Nl, and v_hat for every
     bin/eps), instead of being written out to separate per-bin .npy files.

All paths, filenames, and constants below are otherwise copied verbatim
from the main binned pipeline so they resolve to the same files on disk.
If you change DATA_DIR / NSIDE_MAP / N_REALIZATIONS / SMOOTH_WINDOW /
PATCH_CACHE_TAG in the main pipeline, mirror those changes here too or the
paths won't match.

Design notes (see conversation for full reasoning):
  1. QE is run once PER BIN (v0, v1, v2), each using the full T map (shared)
     + that bin's own tau_b -- the direct generalization of a single-shell
     QE to the 3-tracer case, using the same tracer information (T + all
     three tau_b) that the ViT's 4-channel input encodes.
  2. Cl_TT(eps) = Cl_T_signal + eps^2 * Cl_noise, both ensemble-averaged over
     the 32 training realizations, combined analytically. This exploits
     Tnoise = T + eps*noise (T eps-independent, noise assumed independent of
     signal) so the QE can be evaluated at any eps without new anafast calls.
  3. The QE's filter spans l=10 to l=5000 inclusive (see change #1 above).
     MSE is reported over that same range, l=[10, 5000].
  4. Evaluated only at eps=0.0 (noise-free) and eps=1.0 (full noise), per
     request -- not the full curriculum eps grid.
  5. No patching / NEST reordering anywhere: the QE works directly on
     full-sky RING maps via map2alm/alm2map, matching how the realization
     files are stored on disk (RING order, as written by hp.synfast).
"""

import os
import time
import gc
import numpy as np
import healpy as hp
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##=====================================
##  CONSTANTS -- copied verbatim from the main binned pipeline so file
##  paths match. Edit here ONLY if you also edited them in the main script.
##=====================================
N_REALIZATIONS = 32
N_REALIZATIONS_USED = 64   # <-- must match the ViT job's N_REALIZATIONS_USED
TEST_SEED      = N_REALIZATIONS_USED
NSIDE_MAP      = 2048
LMAX           = 3 * NSIDE_MAP - 1
SMOOTH_WINDOW  = 5

BIN_NAMES = ["0", "1", "2"]

DATA_DIR = "/mnt/beegfs/scoulombe"
OUT_DIR = os.path.join(DATA_DIR, "final_qe_generated")

REAL_FILES = {
    f"v{b}": os.path.join(DATA_DIR, f"realizations_v{b}_nside{NSIDE_MAP}_n{N_REALIZATIONS}_norm_smooth{SMOOTH_WINDOW}.npy")
    for b in BIN_NAMES
}
REAL_FILES.update({
    f"tau{b}": os.path.join(DATA_DIR, f"realizations_tau{b}_nside{NSIDE_MAP}_n{N_REALIZATIONS}_norm_smooth{SMOOTH_WINDOW}.npy")
    for b in BIN_NAMES
})
REAL_FILES["T"] = os.path.join(DATA_DIR, f"realizations_T_nside{NSIDE_MAP}_n{N_REALIZATIONS}.npy")
REAL_FILES["cmbnoise"] = os.path.join(DATA_DIR, f"realizations_cmbnoise_nside{NSIDE_MAP}_n{N_REALIZATIONS}.npy")

TEST_FILES = {
    name: os.path.join(DATA_DIR, f"test_realization_{name}_nside{NSIDE_MAP}_binned_seed{TEST_SEED}.npy")
    for name in ["v0", "v1", "v2", "tau0", "tau1", "tau2", "T", "cmbnoise"]
}

CLS_PATH  = os.path.join(DATA_DIR, f"cls_all_nside{NSIDE_MAP}_lmax{LMAX}_norm.npz")
NORM_PATH = os.path.join(DATA_DIR, f"norm_factors_nside{NSIDE_MAP}.npz")

# Only the two requested scenarios: noise-free and eps=1 (full noise)
QE_EPS_LIST = [0.0, 1.0]

# CHANGED: QE now only uses/reconstructs l=10..5000 (inclusive). This is
# applied directly to the T and tau alms before the estimator's real-space
# multiply (see run_qe_binned), so it constrains the estimator itself, not
# just what range MSE is computed over.
QE_LMIN_FILTER = 10
QE_LMAX_FILTER = 5001  # slice is exclusive at the top, so this reaches 5000 inclusive
QE_FILTER_ARRAY = np.zeros(LMAX + 1)
QE_FILTER_ARRAY[QE_LMIN_FILTER:QE_LMAX_FILTER] = 1.0

# Shared with the full-range run on purpose -- ensemble spectra don't depend
# on the filter, see module docstring change #3.
ENSEMBLE_SPECTRA_CACHE = os.path.join(DATA_DIR, "qe_binned_ensemble_spectra.npz")

# CHANGED: routed into final_qe_generated/ with filter range in the name so
# this never overwrites a previous full-range (or other-range) run.
RESULTS_FILE = os.path.join(OUT_DIR, f"qe_binned_results_l{QE_LMIN_FILTER}-{QE_LMAX_FILTER-1}.npz")
SUMMARY_PLOT = os.path.join(OUT_DIR, f"qe_binned_summary_l{QE_LMIN_FILTER}-{QE_LMAX_FILTER-1}.png")

# If the held-out test realization doesn't exist yet, generate it here with
# the SAME procedure the main pipeline uses (so it matches what the ViT was
# evaluated on). If TEST_FILES already exist (normal case -- you ran the
# main pipeline already), this is skipped and those exact files are used.
GENERATE_TEST_IF_MISSING = True

##=====================================
##  moving_average -- must match the main pipeline exactly
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
##  Reproduce the held-out test realization if missing (identical procedure
##  to the main pipeline's generate_test_realization, so it's the same file
##  the ViT would be/was scored against -- not a fresh/independent draw)
##=====================================
def generate_test_realization_if_missing():
    if all(os.path.exists(p) for p in TEST_FILES.values()):
        print("Test realization files already exist -- using them as-is (same data the ViT used).")
        return
    if not GENERATE_TEST_IF_MISSING:
        raise FileNotFoundError(
            "TEST_FILES are missing and GENERATE_TEST_IF_MISSING=False. "
            "Run the main pipeline first, or set GENERATE_TEST_IF_MISSING=True."
        )

    print(f"TEST_FILES missing -- reproducing seed={TEST_SEED} test realization "
          f"with the same procedure as the main pipeline...")
    import camb

    cls_data = np.load(CLS_PATH)
    field_names = ["tau0", "tau1", "tau2", "v0", "v1", "v2"]
    n_spectra = len(field_names) * (len(field_names) + 1) // 2
    cls_all = [cls_data[str(k)] for k in range(n_spectra)]
    cls_all_smoothed = [moving_average(c, SMOOTH_WINDOW) for c in cls_all]

    np.random.seed(TEST_SEED)
    synth = hp.synfast(cls_all_smoothed, NSIDE_MAP, lmax=LMAX, pol=False, new=True)
    synth = dict(zip(field_names, synth))

    norm_data = np.load(NORM_PATH)
    tau_std0 = float(norm_data["tau_std0"])
    v_std0 = float(norm_data["v_std0"])

    pars = camb.set_params(
        H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0,
        tau=0.06, As=2e-9, ns=0.965, halofit_version='mead', lmax=LMAX,
    )
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit=None, raw_cl=True)
    totCL = powers['total']
    cmb = totCL[:, 0]
    noise_cl = np.array(cmb[:LMAX + 1], dtype=np.float64).copy()
    if len(noise_cl) < LMAX + 1:
        noise_cl = np.pad(noise_cl, (0, LMAX + 1 - len(noise_cl)))
    noise_cl[0] = 0.0
    noise_cl = noise_cl / ((tau_std0 ** 2) * (v_std0 ** 2))

    # NOTE: this reuses TEST_SEED for both the field synth and the noise
    # synth, exactly as the main pipeline does, so the reproduced file is
    # bit-identical to what the ViT was actually scored against. If you
    # separately want to check whether that seed reuse correlates signal
    # and noise, do that as its own investigation -- don't silently change
    # it here, or this stops being "the same data the ViT used."
    np.random.seed(TEST_SEED)
    noise_test = hp.synfast(noise_cl, NSIDE_MAP, lmax=LMAX).astype(np.float32)

    T_test = np.zeros(hp.nside2npix(NSIDE_MAP), dtype=np.float64)
    for b in BIN_NAMES:
        T_test += synth[f"v{b}"].astype(np.float64) * synth[f"tau{b}"].astype(np.float64)
    T_test = T_test.astype(np.float32)

    for b in BIN_NAMES:
        np.save(TEST_FILES[f"v{b}"], synth[f"v{b}"].astype(np.float32))
        np.save(TEST_FILES[f"tau{b}"], synth[f"tau{b}"].astype(np.float32))
    np.save(TEST_FILES["T"], T_test)
    np.save(TEST_FILES["cmbnoise"], noise_test)

    del synth, T_test, noise_test
    gc.collect()
    print("Reproduced and saved test realization files.")

generate_test_realization_if_missing()

##=====================================
##  STEP 1: ensemble (training-set) spectra -- same 32 realizations the ViT
##  trained on, NEVER computed from the test map
##=====================================
def compute_ensemble_spectra(lmax=LMAX):
    if os.path.exists(ENSEMBLE_SPECTRA_CACHE):
        d = np.load(ENSEMBLE_SPECTRA_CACHE)
        Cl_tau_avg = {b: d[f"Cl_tau_{b}"] for b in BIN_NAMES}
        Cl_T_signal_avg = d["Cl_T_signal"]
        Cl_noise_avg = d["Cl_noise"]
        print(f"Loaded ensemble spectra from cache: {ENSEMBLE_SPECTRA_CACHE}")
        return Cl_tau_avg, Cl_T_signal_avg, Cl_noise_avg

    print("Computing ensemble spectra from the 32 TRAINING realizations "
          "(same files REAL_FILES the ViT trained on)...")
    t0 = time.time()

    tau_mmaps = {b: np.load(REAL_FILES[f"tau{b}"], mmap_mode="r") for b in BIN_NAMES}
    T_mmap = np.load(REAL_FILES["T"], mmap_mode="r")
    noise_mmap = np.load(REAL_FILES["cmbnoise"], mmap_mode="r")

    Cl_tau_sum = {b: np.zeros(lmax + 1) for b in BIN_NAMES}
    Cl_T_signal_sum = np.zeros(lmax + 1)
    Cl_noise_sum = np.zeros(lmax + 1)

    for i in range(N_REALIZATIONS):
        for b in BIN_NAMES:
            tau_i = np.array(tau_mmaps[b][i])
            Cl_tau_sum[b] += hp.anafast(tau_i, lmax=lmax)
        T_i = np.array(T_mmap[i])
        Cl_T_signal_sum += hp.anafast(T_i, lmax=lmax)
        noise_i = np.array(noise_mmap[i])
        Cl_noise_sum += hp.anafast(noise_i, lmax=lmax)
        print(f"  ensemble spectra: {i+1}/{N_REALIZATIONS} realizations done")

    Cl_tau_avg = {b: Cl_tau_sum[b] / N_REALIZATIONS for b in BIN_NAMES}
    Cl_T_signal_avg = Cl_T_signal_sum / N_REALIZATIONS
    Cl_noise_avg = Cl_noise_sum / N_REALIZATIONS

    save_kwargs = {f"Cl_tau_{b}": Cl_tau_avg[b] for b in BIN_NAMES}
    save_kwargs["Cl_T_signal"] = Cl_T_signal_avg
    save_kwargs["Cl_noise"] = Cl_noise_avg
    np.savez(ENSEMBLE_SPECTRA_CACHE, **save_kwargs)

    del tau_mmaps, T_mmap, noise_mmap
    gc.collect()
    print(f"Ensemble spectra computed in {time.time()-t0:.1f}s, cached to {ENSEMBLE_SPECTRA_CACHE}")
    return Cl_tau_avg, Cl_T_signal_avg, Cl_noise_avg

Cl_tau_avg, Cl_T_signal_avg, Cl_noise_avg = compute_ensemble_spectra()

def cl_tt_for_eps(eps):
    """Cl_TT(eps) = Cl_T_signal + eps^2 * Cl_noise, both from the training
    ensemble -- lets the QE be evaluated at any eps without new anafast
    calls on the test map."""
    return Cl_T_signal_avg + (eps ** 2) * Cl_noise_avg

##=====================================
##  STEP 2: wigner 3j (needed for the QE's normalization integral)
##=====================================
def wigner_3j_000_torch_batch(l1_grid, l2_grid, l3_scalar, torch_device=device):
    l1 = torch.as_tensor(l1_grid, dtype=torch.float64, device=torch_device)
    l2 = torch.as_tensor(l2_grid, dtype=torch.float64, device=torch_device)
    l3 = torch.tensor(float(l3_scalar), dtype=torch.float64, device=torch_device)

    L = l1 + l2 + l3
    triangle_ok = (torch.abs(l1 - l2) <= l3) & (l1 + l2 >= l3)
    L_int = torch.round(L).long()
    parity_ok = (L_int % 2 == 0)
    valid = triangle_ok & parity_ok

    def logfac(n):
        safe_n = torch.clamp(n, min=0.0)
        result = torch.lgamma(safe_n + 1.0)
        result = torch.where(n < 0, torch.full_like(result, float('-inf')), result)
        return result

    log_sqrt_part = 0.5 * (
        logfac(L - 2 * l1) + logfac(L - 2 * l2) + logfac(L - 2 * l3) - logfac(L + 1)
    )
    halfL = L / 2.0
    log_comb_part = (
        logfac(halfL) - (logfac(halfL - l1) + logfac(halfL - l2) + logfac(halfL - l3))
    )

    finite = torch.isfinite(log_sqrt_part) & torch.isfinite(log_comb_part)
    valid = valid & finite

    halfL_int = torch.round(halfL).long()
    sign = torch.where((halfL_int % 2 == 0), torch.ones_like(L), -torch.ones_like(L))

    value = sign * torch.exp(log_sqrt_part + log_comb_part)
    value = torch.where(valid, value, torch.zeros_like(value))
    return value

def compute_Nl(Cl_tau_b, Cl_TT, lmax=LMAX, lmin_filter=QE_LMIN_FILTER, lmax_filter=QE_LMAX_FILTER):
    """Normalization for a single bin's QE, given that bin's Cl_tau_b and the
    shared Cl_TT(eps). lmin_filter/lmax_filter bound the wigner-3j sum to
    the same l-range the estimator itself is restricted to, so Nl is
    consistent with a windowed (not full-range) QE."""
    inv_Cl_TT = 1.0 / Cl_TT
    bigL = np.round(np.linspace(1, lmax, 25)).astype(int)
    bigLfull = np.arange(lmax + 1)
    Nl_lowres = np.zeros(len(bigL))

    l1 = np.arange(lmin_filter, lmax_filter)[:, None]
    l2 = np.arange(lmin_filter, lmax_filter)[None, :]
    facl1bottom = (Cl_tau_b[lmin_filter:lmax_filter])[:, None]
    facl2 = (inv_Cl_TT[lmin_filter:lmax_filter])[None, :]

    l1_t = torch.as_tensor(l1, dtype=torch.float64, device=device)
    l2_t = torch.as_tensor(l2, dtype=torch.float64, device=device)
    facl1bottom_t = torch.as_tensor(facl1bottom, dtype=torch.float64, device=device)
    facl2_t = torch.as_tensor(facl2, dtype=torch.float64, device=device)

    for i in range(len(bigL)):
        l3 = int(bigL[i])
        W = wigner_3j_000_torch_batch(l1_t, l2_t, l3) ** 2
        Fdown = (2. * l1_t + 1) * (2. * l2_t + 1) * facl1bottom_t * facl2_t
        Sdown = torch.sum(Fdown * W) / 4. / np.pi
        Nl_lowres[i] = (1. / Sdown).item()
        del W, Fdown, Sdown

    return np.interp(bigLfull, bigL, Nl_lowres)

##=====================================
##  STEP 3: per-bin QE reconstruction on the full-sky RING test maps
##=====================================
def run_qe_binned(Tnoise_ring, tau_rings: dict, Cl_tau_avg: dict, Cl_TT: np.ndarray, lmax=LMAX):
    """
    Reconstructs v0, v1, v2 from Tnoise (single full-sky map) + tau_rings =
    {b: tau_b full-sky map} -- the same information content as the ViT's
    4-channel input [Tnoise, tau0, tau1, tau2].

    T_tilde is computed once (shared across bins); each bin then applies its
    own tau_b filtering and its own normalization Nl.

    QE_FILTER_ARRAY is applied to BOTH T_tilde_lm and Tao_tilde_lm via
    almxfl, before the alm2map back to real space. That means l-modes
    outside [QE_LMIN_FILTER, QE_LMAX_FILTER) are exactly zero in
    T_tilde_map and Tao_tilde_map -- the estimator itself only ever uses
    and produces information inside the filter window, not just its score.
    """
    T_lm = hp.map2alm(Tnoise_ring, lmax=lmax, iter=3)
    inv_Cl_TT = 1.0 / Cl_TT
    T_tilde_lm = hp.almxfl(T_lm, inv_Cl_TT * QE_FILTER_ARRAY)
    T_tilde_map = hp.alm2map(T_tilde_lm, NSIDE_MAP)

    def _one_bin(b):
        Tao_lm = hp.map2alm(tau_rings[b], lmax=lmax)
        Tao_tilde_lm = hp.almxfl(Tao_lm, QE_FILTER_ARRAY)
        Tao_tilde_map = hp.alm2map(Tao_tilde_lm, NSIDE_MAP)

        Nl_b = compute_Nl(Cl_tau_avg[b], Cl_TT, lmax=lmax)

        v_hat_lm = hp.map2alm(T_tilde_map * Tao_tilde_map)
        v_hat_lm_N = hp.almxfl(v_hat_lm, Nl_b)
        v_hat_b = hp.alm2map(v_hat_lm_N, nside=NSIDE_MAP)
        return b, v_hat_b, Nl_b

    v_hat = {}
    Nl_dict = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        for b, v_hat_b, Nl_b in ex.map(_one_bin, BIN_NAMES):
            v_hat[b] = v_hat_b
            Nl_dict[b] = Nl_b

    return v_hat, Nl_dict

##=====================================
##  STEP 4: load the held-out test realization (same file, same maps the
##  ViT is evaluated against) and run the QE across the eps grid
##=====================================
v_true_ring   = {b: np.load(TEST_FILES[f"v{b}"]) for b in BIN_NAMES}
tau_ring      = {b: np.load(TEST_FILES[f"tau{b}"]) for b in BIN_NAMES}
T_ring        = np.load(TEST_FILES["T"])
noise_ring    = np.load(TEST_FILES["cmbnoise"])
print(f"Loaded held-out test realization from {DATA_DIR} (seed={TEST_SEED}).")

# CHANGED: two-sided mask matching the filter band exactly (was previously
# ">= QE_LMIN_FILTER" only, which was fine when QE_LMAX_FILTER == LMAX+1,
# but must be bounded on both ends now that the top of the band is 5000).
valid_mask = (np.arange(LMAX + 1) >= QE_LMIN_FILTER) & (np.arange(LMAX + 1) < QE_LMAX_FILTER)

# cl_true doesn't depend on eps -- compute once per bin, not once per (bin, eps)
cl_true_by_bin = {b: hp.anafast(v_true_ring[b], lmax=LMAX) for b in BIN_NAMES}

qe_results = {b: {} for b in BIN_NAMES}

for eps in QE_EPS_LIST:
    print(f"\n=== QE reconstruction at eps={eps} ===")
    Tnoise_ring = (T_ring + eps * noise_ring).astype(np.float32)

    Cl_TT = cl_tt_for_eps(eps)
    t0 = time.time()
    v_hat, Nl_dict = run_qe_binned(Tnoise_ring, tau_ring, Cl_tau_avg, Cl_TT, lmax=LMAX)
    print(f"  QE reconstruction (3 bins) took {time.time()-t0:.1f}s")

    for b in BIN_NAMES:
        cl_true = cl_true_by_bin[b]
        cl_qe = hp.anafast(v_hat[b], lmax=LMAX)
        cl_cross = hp.anafast(v_true_ring[b], v_hat[b], lmax=LMAX)
        mse_qe = np.mean((cl_qe[valid_mask] - cl_true[valid_mask]) ** 2)

        qe_results[b][eps] = {
            "cl_true": cl_true,
            "cl_qe": cl_qe,
            "cl_cross": cl_cross,
            "mse": mse_qe,
            "Nl": Nl_dict[b],
            "v_hat": v_hat[b].astype(np.float32),
        }
        print(f"  bin {b}  eps={eps:<5}  MSE[l={QE_LMIN_FILTER}:{QE_LMAX_FILTER-1}]={mse_qe:.4e}")

##=====================================
##  STEP 5: save results (spectra + Nl + reconstructed map, all in ONE
##  file) + summary plots -- written to final_qe_generated/, never
##  overwriting any previous (e.g. full-range) run
##=====================================
os.makedirs(OUT_DIR, exist_ok=True)

save_kwargs = {}
for b in BIN_NAMES:
    for eps in QE_EPS_LIST:
        save_kwargs[f"cl_true_bin{b}_eps{eps}"] = qe_results[b][eps]["cl_true"]
        save_kwargs[f"cl_qe_bin{b}_eps{eps}"] = qe_results[b][eps]["cl_qe"]
        save_kwargs[f"cl_cross_bin{b}_eps{eps}"] = qe_results[b][eps]["cl_cross"]
        save_kwargs[f"mse_bin{b}_eps{eps}"] = qe_results[b][eps]["mse"]
        save_kwargs[f"Nl_bin{b}_eps{eps}"] = qe_results[b][eps]["Nl"]
        save_kwargs[f"v_hat_bin{b}_eps{eps}"] = qe_results[b][eps]["v_hat"]
np.savez(RESULTS_FILE, **save_kwargs)
print(f"Saved all results (cl_true, cl_qe, cl_cross, Nl, v_hat) to {RESULTS_FILE}")

# One Cl comparison plot per bin: true vs QE(eps=0) vs QE(eps=1), windowed l range
ell = np.arange(LMAX + 1)
fig, axes = plt.subplots(1, len(BIN_NAMES), figsize=(6 * len(BIN_NAMES), 5))
for i, b in enumerate(BIN_NAMES):
    ax = axes[i] if len(BIN_NAMES) > 1 else axes
    ax.plot(ell[valid_mask], qe_results[b][0.0]["cl_true"][valid_mask], color='black', linewidth=2, label="true")
    ax.plot(ell[valid_mask], qe_results[b][0.0]["cl_qe"][valid_mask], label="QE (eps=0, noise-free)")
    ax.plot(ell[valid_mask], qe_results[b][1.0]["cl_qe"][valid_mask], label="QE (eps=1)")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$")
    ax.set_title(f"bin {b}")
    ax.legend()
plt.tight_layout()
plt.savefig(SUMMARY_PLOT, dpi=150)
plt.close(fig)
print(f"Saved summary plot to {SUMMARY_PLOT}")

print("\nDone. This script produced QE-only results (l=10..5000 window) using:")
print(f"  - training spectra from: {list(REAL_FILES.values())}")
print(f"  - test realization from: {list(TEST_FILES.values())}")
print("No ViT model was loaded or evaluated in this script.")