#!/usr/bin/env python3
"""
continuum_gui_app.py  (Streamlit + Plotly)

AGN Continuum Reconstruction & Normalization Toolkit
Version 1.3
Author: Vivek M
Bug reports: vivek.m@iiap.res.in

Adds:
- "Rankine approach" (morphing + MFICA-like reconstruction + Rankine iterative mask),
  driven by a precomputed rankine_bases.npz trained on the SAME wavelength grid as default bases.

IMPORTANT (your masking issue):
- Rankine mask is recomputed *from scratch* each iteration (non-sticky),
  so changing Rankine parameters actually updates (and can UNmask) pixels.

Run:
  pip install streamlit astropy pandas numpy plotly scikit-learn scipy
  streamlit run continuum_gui_app.py
"""

import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
from scipy.ndimage import median_filter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.optimize import nnls
except Exception:
    nnls = None

DEFAULT_BASES_PATH = os.path.join("bases", "default_bases.npz")
DEFAULT_RANKINE_BASES_PATH = os.path.join("bases", "rankine_bases.npz")
C_KMS = 299792.458


# ============================
# Utils
# ============================
def safe_str(x):
    if x is None:
        return "NA"
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("/", "_")
    return s


def get_header_value(hdr, keys, default=None):
    for k in keys:
        if k in hdr:
            return hdr[k]
    return default


def contiguous_mask(mask_bool, min_run=3, grow=1):
    """Enforce min_run contiguous True pixels and grow by +/-grow."""
    m = mask_bool.astype(bool)
    if not np.any(m):
        return m

    idx = np.where(m)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]

    out = np.zeros_like(m, dtype=bool)
    for s, e in zip(starts, ends):
        if (e - s + 1) >= int(min_run):
            ss = max(0, s - int(grow))
            ee = min(len(m) - 1, e + int(grow))
            out[ss:ee + 1] = True
    return out


def interp_to_grid(w_in, y_in, w_out):
    return np.interp(w_out, w_in, y_in, left=np.nan, right=np.nan)


def interp_ivar_to_grid(w_in, iv_in, w_out):
    iv = np.interp(w_out, w_in, iv_in, left=0.0, right=0.0)
    return np.maximum(iv, 0.0)


def intervals_to_mask(wave, intervals):
    m = np.zeros_like(wave, dtype=bool)
    for lo, hi in intervals:
        if lo is None or hi is None:
            continue
        lo2 = float(min(lo, hi))
        hi2 = float(max(lo, hi))
        m |= (wave >= lo2) & (wave <= hi2)
    return m


def mask_to_segments(mask_bool):
    m = mask_bool.astype(bool)
    if not np.any(m):
        return []
    idx = np.where(m)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))


def median_scale_spectrum(flux, trusted):
    if np.any(trusted):
        med = np.median(flux[trusted])
        if np.isfinite(med) and med > 0:
            return float(med)
    return 1.0


# ============================
# FITS reading + z extraction
# ============================
def extract_z_from_hdu2(hdul):
    try:
        if len(hdul) <= 2:
            return None
        d2 = getattr(hdul[2], "data", None)
        if d2 is None:
            return None

        if hasattr(d2, "dtype") and d2.dtype is not None and d2.dtype.names is not None and "Z" in d2.dtype.names:
            zcand = np.array(d2["Z"]).ravel()
            zcand = zcand[np.isfinite(zcand)]
            if zcand.size > 0 and float(zcand[0]) > 0:
                return float(zcand[0])

        if hasattr(d2, "__getitem__"):
            try:
                zcand = np.array(d2["Z"]).ravel()
                zcand = zcand[np.isfinite(zcand)]
                if zcand.size > 0 and float(zcand[0]) > 0:
                    return float(zcand[0])
            except Exception:
                pass
    except Exception:
        return None
    return None


def read_sdss_fits(uploaded_file_bytes):
    """
    Supports:
      A) table with flux/loglam/ivar
      B) 1D image with CRVAL1/CD1_1
    """
    hdul = fits.open(io.BytesIO(uploaded_file_bytes))
    hdr0 = hdul[0].header

    z_from_hdu2 = extract_z_from_hdu2(hdul)

    plate = get_header_value(hdr0, ["PLATE", "PLATEID"], None)
    mjd = get_header_value(hdr0, ["MJD"], None)
    fiberid = get_header_value(hdr0, ["FIBERID", "FIBER"], None)
    ra = get_header_value(hdr0, ["RA", "OBJRA", "PLUG_RA"], None)
    dec = get_header_value(hdr0, ["DEC", "OBJDEC", "PLUG_DEC"], None)
    meta = dict(plate=plate, mjd=mjd, fiberid=fiberid, ra=ra, dec=dec)

    # Case A: table
    for hdu in hdul[1:]:
        if getattr(hdu, "data", None) is None or not hasattr(hdu.data, "columns"):
            continue
        cols = [c.name.lower() for c in hdu.data.columns]
        if ("flux" in cols) and (("loglam" in cols) or ("wavelength" in cols) or ("lam" in cols)):
            data = hdu.data

            flux = np.array(data["flux"], dtype=np.float64).squeeze()
            if flux.ndim > 1:
                flux = flux[0]

            if "ivar" in cols:
                ivar = np.array(data["ivar"], dtype=np.float64).squeeze()
            elif "invvar" in cols:
                ivar = np.array(data["invvar"], dtype=np.float64).squeeze()
            else:
                ivar = np.ones_like(flux, dtype=np.float64)
            if ivar.ndim > 1:
                ivar = ivar[0]

            if "loglam" in cols:
                loglam = np.array(data["loglam"], dtype=np.float64).squeeze()
                if loglam.ndim > 1:
                    loglam = loglam[0]
                wave_obs = 10 ** loglam
            elif "wavelength" in cols:
                wave_obs = np.array(data["wavelength"], dtype=np.float64).squeeze()
                if wave_obs.ndim > 1:
                    wave_obs = wave_obs[0]
            else:
                wave_obs = np.array(data["lam"], dtype=np.float64).squeeze()
                if wave_obs.ndim > 1:
                    wave_obs = wave_obs[0]

            hdul.close()
            return wave_obs, flux, ivar, hdr0, z_from_hdu2, meta

    # Case B: 1D image
    flux_img = np.array(hdul[0].data, dtype=np.float64).squeeze()
    if flux_img.ndim != 1:
        hdul.close()
        raise ValueError("Unsupported FITS: could not find 1D spectrum table or 1D image.")

    crval1 = get_header_value(hdr0, ["CRVAL1"], None)
    cd1_1 = get_header_value(hdr0, ["CD1_1", "CDELT1"], None)
    crpix1 = get_header_value(hdr0, ["CRPIX1"], 1.0)
    if crval1 is None or cd1_1 is None:
        hdul.close()
        raise ValueError("Could not infer wavelength: header missing CRVAL1 and CD1_1/CDELT1.")

    pix = np.arange(len(flux_img), dtype=np.float64)
    wave_obs = crval1 + (pix + 1 - float(crpix1)) * float(cd1_1)
    ivar = np.ones_like(flux_img, dtype=np.float64)
    hdul.close()
    return wave_obs, flux_img, ivar, hdr0, z_from_hdu2, meta


# ============================
# Bases loading (cached)
# ============================
@st.cache_resource(show_spinner=False)
def load_npz_from_path(path: str):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


@st.cache_resource(show_spinner=False)
def load_npz_from_bytes(b: bytes):
    d = np.load(io.BytesIO(b), allow_pickle=False)
    return {k: d[k] for k in d.files}


# ============================
# PCA/ICA/EMPCA/NMF helpers
# ============================
def bases_trained_with_robust_scaler(bases):
    flag = bases.get("trained_with_robust_scaler", np.array([0], dtype=np.int8))
    return int(flag[0]) == 1


def apply_training_robust_scaler(X, bases):
    if not bases_trained_with_robust_scaler(bases):
        return X, None
    center = bases.get("scaler_center", None)
    scale = bases.get("scaler_scale", None)
    if center is None or scale is None or center.size == 0 or scale.size == 0:
        return X, None
    Xs = (X - center[None, :]) / scale[None, :]
    inv = (center, scale)
    return Xs, inv


def inverse_training_robust_scaler(Xs, inv):
    if inv is None:
        return Xs
    center, scale = inv
    return Xs * scale[None, :] + center[None, :]


def solve_coeffs_weighted_ls(V, y, w):
    ww = np.sqrt(np.maximum(w, 0.0))
    A = (V.T * ww[:, None])  # LxK
    b = y * ww
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    return a


def solve_coeffs_weighted_nnls(H, y, w):
    K, _ = H.shape
    ww = np.sqrt(np.maximum(w, 0.0))
    A = (H.T * ww[:, None])  # LxK
    b = y * ww

    if nnls is not None:
        a, _ = nnls(A, b)
        return a

    # fallback projected GD
    a = np.maximum(0.0, np.random.rand(K))
    AtA = A.T @ A
    Atb = A.T @ b
    Llip = np.linalg.norm(AtA, 2) + 1e-12
    lr = 1.0 / Llip
    for _ in range(800):
        grad = AtA @ a - Atb
        a = np.maximum(0.0, a - lr * grad)
    return a


def reconstruct_continuum(method, bases, flux_feat, ivar_scaled, trusted_fit):
    w = ivar_scaled * trusted_fit

    if method == "PCA":
        mu = bases["pca_mu"].astype(np.float64)
        V = bases["pca_V"].astype(np.float64)
        y = flux_feat - mu
        a = solve_coeffs_weighted_ls(V, y, w)
        return mu + a @ V

    if method == "ICA":
        mu = bases["ica_mu"].astype(np.float64)
        V = bases["ica_V"].astype(np.float64)
        y = flux_feat - mu
        a = solve_coeffs_weighted_ls(V, y, w)
        return mu + a @ V

    if method == "EMPCA":
        mu = bases["empca_mu"].astype(np.float64)
        V = bases["empca_V"].astype(np.float64)
        y = flux_feat - mu
        a = solve_coeffs_weighted_ls(V, y, w)
        return mu + a @ V

    if method == "NMF":
        H = bases["nmf_H"].astype(np.float64)  # KxL
        shift = bases["nmf_shift"].astype(np.float64)  # L
        y = (flux_feat + shift)
        a = solve_coeffs_weighted_nnls(H, y, w)
        cont = (a @ H) - shift
        return cont

    raise ValueError(f"Unknown method: {method}")


def compute_metrics(flux_s, cont, ivar_s, good_mask):
    good = good_mask & np.isfinite(flux_s) & np.isfinite(cont) & (ivar_s > 0)
    n = int(np.sum(good))
    if n < 5:
        return dict(N=n, chi2_nu=np.nan, wRMSE=np.nan, med_abs_z=np.nan)
    r = flux_s[good] - cont[good]
    w = ivar_s[good]
    chi2 = float(np.sum(r * r * w))
    dof = max(n - 1, 1)
    chi2_nu = chi2 / dof
    wsum = float(np.sum(w))
    wRMSE = np.sqrt(float(np.sum(w * r * r)) / max(wsum, 1e-30))
    z = r * np.sqrt(w)
    med_abs_z = float(np.median(np.abs(z)))
    return dict(N=n, chi2_nu=chi2_nu, wRMSE=wRMSE, med_abs_z=med_abs_z)


# ============================
# Rankine approach helpers
# ============================
def initial_rankine_windows_mask(wave):
    m = np.zeros_like(wave, dtype=bool)
    m |= (wave >= 1295) & (wave <= 1400)   # SiIV+OIV
    m |= (wave >= 1430) & (wave <= 1546)   # CIV
    m |= (wave >= 1780) & (wave <= 1880)   # AlIII
    return m


def narrow_absorption_mask(flux, ivar, medwin=61, nsig=3.0, grow=3):
    good = (ivar > 0) & np.isfinite(flux)
    if not np.any(good):
        return np.zeros_like(flux, dtype=bool)
    fill = np.nanmedian(flux[good])
    pseudo = median_filter(np.where(good, flux, fill), size=int(medwin), mode="nearest")
    sigma = np.zeros_like(flux)
    sigma[good] = 1.0 / np.sqrt(ivar[good])
    m = np.zeros_like(flux, dtype=bool)
    m[good] = (flux[good] < (pseudo[good] - float(nsig) * sigma[good]))
    return contiguous_mask(m, min_run=1, grow=int(grow))


def weighted_ls_recon(mu, V, flux, ivar, fit_mask):
    """
    mu: (L,)
    V : (K,L)
    recon = mu + a@V
    """
    w = ivar * fit_mask.astype(np.float64)
    y = flux - mu
    a = solve_coeffs_weighted_ls(V, y, w)
    recon = mu + a @ V
    return recon, a


def interpolate_over_emission_lines(wave, y, line_windows):
    yy = y.copy()
    mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in line_windows:
        mask |= (wave >= lo) & (wave <= hi)
    good = (~mask) & np.isfinite(yy)
    if np.sum(good) < 10:
        return yy
    yy[mask] = np.interp(wave[mask], wave[good], yy[good])
    return yy


def civ_mask_to_siiv_velocity_map(wave, civ_mask, civ_lambda0=1549.06, siiv_lambda0=1400.0):
    out = np.zeros_like(civ_mask, dtype=bool)
    idx = np.where(civ_mask)[0]
    if idx.size == 0:
        return out
    lam_civ = wave[idx]
    v = C_KMS * (lam_civ / civ_lambda0 - 1.0)
    lam_siiv = siiv_lambda0 * (1.0 + v / C_KMS)
    j = np.searchsorted(wave, lam_siiv)
    j = np.clip(j, 0, wave.size - 1)
    out[j] = True
    out[np.clip(j - 1, 0, wave.size - 1)] = True
    out[np.clip(j + 1, 0, wave.size - 1)] = True
    return out


def rankine_iterative_mask_nonsticky(
    wave, flux, ivar, recon,
    always_mask,
    N_sigma=2.5,
    half_window=30,
    majority=0.65,
    grow_pixels=10,
    restrict_to_bal_windows=True,
    protect_line_cores=False,
    protect_core_halfwidth_A=10.0,
):
    """
    Non-sticky Rankine iterative mask (fixes your "mask always broad" issue):
    - Compute NEW mask from recon (one-sided: recon - flux > N*sigma) using majority vote
    - Optionally restrict voting to BAL-prior windows
    - Grow trough regions by grow_pixels
    - Apply CIV->SiIV mapping (step v) based on CIV part of the new mask
    Returns:
      iter_mask (does NOT include always_mask)
    """
    valid = (ivar > 0) & np.isfinite(flux) & np.isfinite(recon)
    sigma = np.zeros_like(flux)
    sigma[valid] = 1.0 / np.sqrt(ivar[valid])

    resid = recon - flux   # positive for absorption dips
    cond = np.zeros_like(flux, dtype=bool)
    cond[valid] = resid[valid] > (float(N_sigma) * sigma[valid])

    # Optional: protect emission line cores from being flagged
    if protect_line_cores:
        # CIV ~1549, SiIV+OIV ~1400, AlIII ~1857 (blend)
        cores = [1549.06, 1400.0, 1857.0]
        for c0 in cores:
            cond[(wave >= (c0 - protect_core_halfwidth_A)) & (wave <= (c0 + protect_core_halfwidth_A))] = False

    # Restrict *voting* region
    if restrict_to_bal_windows:
        prior = initial_rankine_windows_mask(wave)
        cond = cond & prior

    # Majority vote in +/-half_window pixels
    L = flux.size
    x = cond.astype(np.int32)
    csum = np.r_[0, np.cumsum(x)]
    iter_mask = np.zeros_like(cond, dtype=bool)

    hw = int(half_window)
    maj = float(majority)
    for i in range(L):
        lo = max(0, i - hw)
        hi = min(L - 1, i + hw)
        cnt = csum[hi + 1] - csum[lo]
        win = hi - lo + 1
        if cnt >= int(np.ceil(maj * win)):
            iter_mask[i] = True

    # Grow each region (step iv)
    iter_mask = contiguous_mask(iter_mask, min_run=1, grow=int(grow_pixels))

    # Step (v): CIV velocity map -> SiIV region
    civ_region = (wave >= 1430) & (wave <= 1546)
    civ_mask = iter_mask & civ_region
    iter_mask |= civ_mask_to_siiv_velocity_map(wave, civ_mask)

    # Ensure we never "unmask" always_mask—handled outside by union,
    # but also avoid creating mask on invalid pixels
    iter_mask &= (~always_mask)
    iter_mask &= (ivar > 0)

    return iter_mask


def morph_spectrum_rankine(
    wave, flux, ivar,
    mu_pre, V_pre,
    ref_sed,
    morph_medfilt_pix=601,
    morph_ratio_lo=0.7,
    morph_ratio_hi=1.6,
    morph_edge_guard_pix=10,
    line_windows=None,
):
    """
    Rankine morphing proxy:
      - do a preliminary reconstruction
      - compute ratio ref / recon_cont (with emission-line interpolation)
      - median filter ratio, clip ratio, guard edges
      - apply multiplicative correction to flux and ivar
    """
    if line_windows is None:
        line_windows = [(1385, 1415), (1500, 1600), (1850, 1960), (2700, 2900)]

    L = wave.size
    valid = (ivar > 0) & np.isfinite(flux)
    edge_guard = np.zeros(L, dtype=bool)
    g = int(morph_edge_guard_pix)
    if g > 0 and 2 * g < L:
        edge_guard[:g] = True
        edge_guard[-g:] = True

    # narrow absorber mask (always)
    m_narrow = narrow_absorption_mask(flux, ivar, medwin=61, nsig=3.0, grow=3)

    # initial BAL-prior windows are masked for this *prelim* morphing fit (like paper intent)
    m_init = initial_rankine_windows_mask(wave)

    fit_mask = valid & (~m_narrow) & (~m_init) & (~edge_guard)
    recon, _ = weighted_ls_recon(mu_pre, V_pre, flux, ivar, fit_mask)

    cont_recon = interpolate_over_emission_lines(wave, recon, line_windows)
    cont_ref = interpolate_over_emission_lines(wave, ref_sed, line_windows)

    eps = 1e-12
    ratio = cont_ref / np.maximum(cont_recon, eps)

    # clip extreme ratios (prevents edge spikes + insane morphing)
    ratio = np.clip(ratio, float(morph_ratio_lo), float(morph_ratio_hi))

    smooth = median_filter(ratio, size=int(morph_medfilt_pix), mode="nearest")

    # guard edges: force correction=1 (prevents boundary spikes)
    if np.any(edge_guard):
        smooth[edge_guard] = 1.0

    flux_m = flux * smooth
    ivar_m = ivar / np.maximum(smooth, eps) ** 2

    # optionally kill ivar on guarded edges for safety downstream
    if np.any(edge_guard):
        ivar_m[edge_guard] = 0.0

    return flux_m, ivar_m, smooth


def rankine_continuum_single(
    wave, flux, ivar,
    rank_bases,
    # morph params
    morph_medfilt_pix=601,
    morph_ratio_lo=0.7,
    morph_ratio_hi=1.6,
    morph_edge_guard_pix=10,
    # mask params
    n_iter=12,
    N_sigma=2.5,
    majority=0.65,
    half_window=30,
    grow_pixels=10,
    restrict_to_bal_windows=True,
    protect_line_cores=False,
    protect_core_halfwidth_A=10.0,
    # external masks
    user_mask=None,
):
    """
    Full Rankine approach on one spectrum:
      - morph (using pre components + ref_sed)
      - iterative mask (non-sticky, recomputed each iteration)
      - final fit using definitive components
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    ivar = np.asarray(ivar, dtype=np.float64)

    # required keys in rank_bases
    mu_pre = rank_bases["mu_pre"].astype(np.float64)
    V_pre = rank_bases["V_pre"].astype(np.float64)
    mu_def = rank_bases["mu_def"].astype(np.float64)
    V_def = rank_bases["V_def"].astype(np.float64)
    ref_sed = rank_bases["ref_sed"].astype(np.float64)

    # morph
    flux_m, ivar_m, smooth = morph_spectrum_rankine(
        wave, flux, ivar,
        mu_pre=mu_pre, V_pre=V_pre, ref_sed=ref_sed,
        morph_medfilt_pix=morph_medfilt_pix,
        morph_ratio_lo=morph_ratio_lo,
        morph_ratio_hi=morph_ratio_hi,
        morph_edge_guard_pix=morph_edge_guard_pix,
    )

    valid = (ivar_m > 0) & np.isfinite(flux_m)

    # always-mask: invalid + narrow absorbers + user masks
    always_mask = np.zeros_like(valid, dtype=bool)
    always_mask |= (~valid)
    always_mask |= narrow_absorption_mask(flux_m, ivar_m, medwin=61, nsig=3.0, grow=3)
    if user_mask is not None:
        always_mask |= user_mask.astype(bool)

    # start mask: include initial prior windows, but DO NOT make them permanently sticky
    mask = always_mask | initial_rankine_windows_mask(wave)

    # iterate: recompute iterative part from scratch each time
    for _ in range(int(n_iter)):
        fit_mask = valid & (~mask)
        recon, _ = weighted_ls_recon(mu_def, V_def, flux_m, ivar_m, fit_mask)

        iter_mask = rankine_iterative_mask_nonsticky(
            wave=wave,
            flux=flux_m,
            ivar=ivar_m,
            recon=recon,
            always_mask=always_mask,
            N_sigma=N_sigma,
            half_window=half_window,
            majority=majority,
            grow_pixels=grow_pixels,
            restrict_to_bal_windows=restrict_to_bal_windows,
            protect_line_cores=protect_line_cores,
            protect_core_halfwidth_A=protect_core_halfwidth_A,
        )

        # new mask = always + initial windows + iterative (fresh)
        mask = always_mask | initial_rankine_windows_mask(wave) | iter_mask

    # final fit
    fit_mask = valid & (~mask)
    cont, _ = weighted_ls_recon(mu_def, V_def, flux_m, ivar_m, fit_mask)

    return dict(
        flux_m=flux_m,
        ivar_m=ivar_m,
        cont=cont,
        mask=mask,
        morph_smooth=smooth,
    )


# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="AGNContinuumNorm v1.3", layout="wide")

st.markdown("### AGN Continuum Reconstruction & Normalization Toolkit")
st.caption("Version 1.3")

with st.expander("About this app"):
    st.markdown(
        """
**Author:** Vivek M  
**Institution:** Indian Institute of Astrophysics  

Methods:
- PCA, ICA, EMPCA, NMF (basis NPZ: `bases/default_bases.npz`)
- **Rankine approach** (morphing + MFICA-like ICA proxy + Rankine iterative mask; basis NPZ: `bases/rankine_bases.npz`)

📩 Bug reports / feature requests: **vivek.m@iiap.res.in**
"""
    )
st.markdown("---")

with st.sidebar:
    st.header("Inputs")

    bases_override = st.file_uploader("Optional: Override default basis NPZ", type=["npz"])
    rankine_override = st.file_uploader("Optional: Override Rankine basis NPZ", type=["npz"])

    spec_file = st.file_uploader("Upload SDSS FITS spectrum", type=["fits", "fit", "fz"])

    method = st.selectbox(
        "Methodology (for plot & download)",
        ["PCA", "ICA", "EMPCA", "NMF", "Rankine approach"],
        index=2
    )

    st.subheader("Rest-frame redshift")
    z_user = st.number_input(
        "Redshift z (if >0, overrides file-derived redshift)",
        value=0.0, step=0.001, format="%.6f"
    )

    st.subheader("Scaling")
    use_median_scale = st.checkbox("Per-spectrum median scaling BEFORE fitting (recommended)", value=True)

    st.subheader("User-defined masks")
    mask_frame = st.radio(
        "Manual mask wavelength frame",
        ["Rest-frame (Å)", "Observed-frame (Å)"],
        index=0
    )

    if "user_masks" not in st.session_state:
        st.session_state.user_masks = []

    with st.expander("Add / manage masks", expanded=False):
        colA, colB = st.columns(2)
        lo = colA.number_input("Mask λ_low", value=0.0, step=1.0, format="%.2f")
        hi = colB.number_input("Mask λ_high", value=0.0, step=1.0, format="%.2f")

        c1, c2, c3 = st.columns(3)
        if c1.button("Add mask"):
            if lo > 0 and hi > 0 and hi != lo:
                st.session_state.user_masks.append((float(min(lo, hi)), float(max(lo, hi))))
            else:
                st.warning("Provide λ_low and λ_high (both > 0 and not equal).")

        if c2.button("Remove last") and len(st.session_state.user_masks) > 0:
            st.session_state.user_masks.pop()

        if c3.button("Clear all"):
            st.session_state.user_masks = []

        if len(st.session_state.user_masks) == 0:
            st.caption("No user masks yet.")
        else:
            st.write("Current masks (in selected frame):")
            st.table(pd.DataFrame(st.session_state.user_masks, columns=["λ_low", "λ_high"]))

    st.subheader("Flux scale option")
    apply_flambda_rest = st.checkbox(
        "Apply strict f_lambda rest scaling: f_rest=(1+z)f_obs, ivar_rest=ivar_obs/(1+z)^2",
        value=False
    )

    show_masked_as_gaps = st.checkbox("Hide masked pixels in plot (set to NaN)", value=False)

    # Default-method mask settings
    st.subheader("Default mask settings (PCA/ICA/EMPCA/NMF)")
    k_def = st.slider("k (sigma): mask if flux < cont - k*sigma", 1.0, 6.0, 2.0, 0.1)
    min_run_def = st.slider("min_run (pixels)", 1, 20, 3, 1)
    grow_def = st.slider("grow (pixels)", 0, 15, 1, 1)

    # Rankine settings
    st.subheader("Rankine Approach mask refinement")
    rank_n_iter = st.slider("Rankine iterations", 3, 20, 12, 1)
    rank_N = st.slider("Rankine N (sigma threshold)", 1.5, 5.0, 2.5, 0.1)
    rank_majority = st.slider("Rankine majority fraction", 0.50, 0.85, 0.65, 0.01)
    rank_halfwin = st.slider("Rankine half-window (pixels)", 10, 60, 30, 1)
    rank_grow = st.slider("Rankine grow (pixels)", 0, 25, 10, 1)
    rank_restrict = st.checkbox("Restrict Rankine voting to BAL windows", value=True)
    rank_protect = st.checkbox("Protect emission-line cores from masking", value=True)
    rank_protect_hw = st.slider("Core protection halfwidth (Å)", 2.0, 30.0, 10.0, 0.5)

    st.subheader("Morphing parameters")
    morph_medfilt = st.slider("Morph median-filter size (pixels)", 101, 2001, 601, 50)
    morph_ratio_lo = st.slider("Morph ratio clip low", 0.2, 1.0, 0.7, 0.05)
    morph_ratio_hi = st.slider("Morph ratio clip high", 1.0, 3.0, 1.6, 0.05)
    morph_edge_guard = st.slider("Morph edge guard (pixels)", 0, 400, 10, 10)

    run_btn = st.button("Reconstruct + Plot", type="primary")

# ---- Require spectrum ----
if spec_file is None:
    st.info("Upload an SDSS FITS spectrum to begin.")
    st.stop()

# ---- Load default bases ----
try:
    bases = load_npz_from_path(DEFAULT_BASES_PATH)
    bases_source = f"default: {DEFAULT_BASES_PATH}"
except Exception as e:
    st.error(f"Failed to load default bases from {DEFAULT_BASES_PATH}: {e}")
    st.stop()

if bases_override is not None:
    try:
        bases = load_npz_from_bytes(bases_override.getvalue())
        bases_source = "override upload"
    except Exception as e:
        st.error(f"Failed to load uploaded default bases NPZ: {e}")
        st.stop()

# ---- Load Rankine bases (optional; may be unavailable or mismatched) ----
rankine_ok = False
rank_bases = None
rank_source = None
try:
    if rankine_override is not None:
        rank_bases = load_npz_from_bytes(rankine_override.getvalue())
        rank_source = "override upload"
    else:
        if os.path.exists(DEFAULT_RANKINE_BASES_PATH):
            rank_bases = load_npz_from_path(DEFAULT_RANKINE_BASES_PATH)
            rank_source = f"default: {DEFAULT_RANKINE_BASES_PATH}"
except Exception:
    rank_bases = None

# ---- Load spectrum + z ----
try:
    wave_obs, flux_obs, ivar_obs, hdr0, z_from_hdu2, meta = read_sdss_fits(spec_file.getvalue())
except Exception as e:
    st.error(f"Failed to read FITS: {e}")
    st.stop()

plate = meta.get("plate", None)
mjd = meta.get("mjd", None)
fiberid = meta.get("fiberid", None)
pmf = f"{safe_str(plate)}-{safe_str(mjd)}-{safe_str(fiberid)}"

z_hdr = get_header_value(hdr0, ["Z", "REDSHIFT"], None)
if float(z_user) > 0:
    z = float(z_user)
    z_source = "user"
elif z_from_hdu2 is not None and np.isfinite(z_from_hdu2) and z_from_hdu2 > 0:
    z = float(z_from_hdu2)
    z_source = "HDU2.Z"
elif z_hdr is not None and isinstance(z_hdr, (float, int)) and np.isfinite(z_hdr) and float(z_hdr) > 0:
    z = float(z_hdr)
    z_source = "HDR"
else:
    z = 0.0
    z_source = "default(0)"

st.write(f"Using redshift z = {z:.6f} (source: {z_source})")

# observed -> rest wavelength
wave_rest_in = wave_obs / (1.0 + z) if (z is not None and z >= 0) else wave_obs

# Optional strict f_lambda rest scaling
flux_in = flux_obs.astype(np.float64).copy()
ivar_in = ivar_obs.astype(np.float64).copy()
if apply_flambda_rest and (z is not None and z >= 0):
    flux_in = flux_in * (1.0 + z)
    ivar_in = ivar_in / ((1.0 + z) ** 2)

# Interpolate onto model REST-FRAME grid (default bases grid)
wave_train = bases["wave"].astype(np.float64)
wmin_in, wmax_in = float(np.nanmin(wave_rest_in)), float(np.nanmax(wave_rest_in))
wmin_tr, wmax_tr = float(np.nanmin(wave_train)), float(np.nanmax(wave_train))
if min(wmax_in, wmax_tr) <= max(wmin_in, wmin_tr):
    st.error(
        f"No wavelength overlap after rest-frame conversion.\n"
        f"Input rest λ range: [{wmin_in:.1f}, {wmax_in:.1f}] Å\n"
        f"Model grid range:   [{wmin_tr:.1f}, {wmax_tr:.1f}] Å"
    )
    st.stop()

flux_g = interp_to_grid(wave_rest_in, flux_in, wave_train)
ivar_g = interp_ivar_to_grid(wave_rest_in, ivar_in, wave_train)

valid_g = np.isfinite(flux_g) & (ivar_g > 0)
flux_g = np.where(np.isfinite(flux_g), flux_g, 0.0)
wave_rest = wave_train

# ---- Build USER mask on rest grid ----
if mask_frame.startswith("Rest"):
    user_mask = intervals_to_mask(wave_rest, st.session_state.user_masks)
    user_mask_note = "manual masks applied in REST frame"
else:
    intervals_rest = []
    for lo, hi in st.session_state.user_masks:
        if lo is None or hi is None:
            continue
        lo2, hi2 = float(min(lo, hi)), float(max(lo, hi))
        if (z is not None) and (z >= 0):
            lo2 /= (1.0 + z)
            hi2 /= (1.0 + z)
        intervals_rest.append((lo2, hi2))
    user_mask = intervals_to_mask(wave_rest, intervals_rest)
    user_mask_note = "manual masks provided in OBS frame, converted to REST"

st.caption(f"Manual mask diagnostic: masked pixels = {int(user_mask.sum())}/{user_mask.size} ({user_mask_note})")

# ---- Check Rankine bases compatibility with this grid ----
if rank_bases is not None and ("wave" in rank_bases):
    rw = np.asarray(rank_bases["wave"]).squeeze()
    if rw.shape == wave_rest.shape and np.max(np.abs(rw - wave_rest)) <= 1e-8:
        required = ["mu_pre", "V_pre", "mu_def", "V_def", "ref_sed"]
        if all(k in rank_bases for k in required):
            rankine_ok = True

if rank_bases is not None:
    if rankine_ok:
        st.sidebar.success(f"Rankine bases loaded: **{rank_source}**")
    else:
        st.sidebar.warning(
            "Rankine bases loaded but NOT usable.\n"
            "Either wavelength grid differs from default bases grid or required keys are missing.\n"
            "Regenerate rankine_bases.npz on the same grid (and include mu_pre/V_pre/mu_def/V_def/ref_sed)."
        )
else:
    st.sidebar.info("Rankine bases not found (optional). Place at bases/rankine_bases.npz or upload one.")

st.sidebar.info(f"Default bases: **{bases_source}**")

# If Rankine chosen but not OK -> force skip
if method == "Rankine approach" and not rankine_ok:
    st.error("Rankine bases wavelength grid does not match the default bases grid (or keys missing). Rankine approach will be skipped.")
    st.stop()

if not run_btn:
    st.stop()

# ============================
# Shared scaling
# ============================
trusted_fit_basic = valid_g & (ivar_g > 0) & (~user_mask)

scale = 1.0
if use_median_scale:
    scale = median_scale_spectrum(flux_g, trusted_fit_basic)

flux_s = flux_g / scale
ivar_s = ivar_g * (scale ** 2)

sigma = np.zeros_like(ivar_s, dtype=np.float64)
good_ivar = ivar_s > 0
sigma[good_ivar] = 1.0 / np.sqrt(ivar_s[good_ivar])

# ============================
# Compute continua
# ============================
all_methods_default = ["PCA", "ICA", "EMPCA", "NMF"]
results = {}

# ----- Default methods -----
for mth in all_methods_default:
    if mth in ["PCA", "ICA", "NMF"]:
        Xfeat_2d = flux_s[None, :]
        Xfeat_s_2d, inv_scaler = apply_training_robust_scaler(Xfeat_2d, bases)
        flux_feat = Xfeat_s_2d[0]
    else:
        inv_scaler = None
        flux_feat = flux_s

    # initial fit (excluding user masks only)
    cont0_feat = reconstruct_continuum(mth, bases, flux_feat, ivar_s, trusted_fit_basic)
    if mth in ["PCA", "ICA", "NMF"]:
        cont0 = inverse_training_robust_scaler(cont0_feat[None, :], inv_scaler)[0]
    else:
        cont0 = cont0_feat

    # default one-sided absorption mask
    mask_raw = np.zeros_like(flux_s, dtype=bool)
    mask_raw[trusted_fit_basic] = (flux_s[trusted_fit_basic] < (cont0[trusted_fit_basic] - float(k_def) * sigma[trusted_fit_basic]))
    mask_auto = contiguous_mask(mask_raw, min_run=min_run_def, grow=grow_def)

    mask_total = mask_auto | user_mask

    # refit excluding total mask
    trusted_fit2 = valid_g & (ivar_g > 0) & (~mask_total)
    cont_feat = reconstruct_continuum(mth, bases, flux_feat, ivar_s, trusted_fit2)
    if mth in ["PCA", "ICA", "NMF"]:
        cont = inverse_training_robust_scaler(cont_feat[None, :], inv_scaler)[0]
    else:
        cont = cont_feat

    good_eval = (ivar_s > 0) & np.isfinite(flux_s) & np.isfinite(cont) & (~mask_total)
    metrics = compute_metrics(flux_s, cont, ivar_s, good_eval)

    results[mth] = dict(cont=cont, mask_total=mask_total, metrics=metrics)

# ----- Rankine approach -----
if rankine_ok:
    # Rankine must see the same scaling as the default methods for fair comparisons,
    # so we feed flux_s/ivar_s and then treat its output as "scaled flux".
    # Also, we union user_mask into Rankine "always_mask".
    rank_out = rankine_continuum_single(
        wave=wave_rest,
        flux=flux_s,
        ivar=ivar_s,
        rank_bases=rank_bases,
        morph_medfilt_pix=morph_medfilt,
        morph_ratio_lo=morph_ratio_lo,
        morph_ratio_hi=morph_ratio_hi,
        morph_edge_guard_pix=morph_edge_guard,
        n_iter=rank_n_iter,
        N_sigma=rank_N,
        majority=rank_majority,
        half_window=rank_halfwin,
        grow_pixels=rank_grow,
        restrict_to_bal_windows=rank_restrict,
        protect_line_cores=rank_protect,
        protect_core_halfwidth_A=rank_protect_hw,
        user_mask=user_mask,
    )

    cont_r = rank_out["cont"]
    mask_r = rank_out["mask"]
    flux_r = rank_out["flux_m"]      # morphed, in the SAME scaling space
    ivar_r = rank_out["ivar_m"]

    good_eval_r = (ivar_r > 0) & np.isfinite(flux_r) & np.isfinite(cont_r) & (~mask_r)
    metrics_r = compute_metrics(flux_r, cont_r, ivar_r, good_eval_r)

    results["Rankine approach"] = dict(
        cont=cont_r,
        mask_total=mask_r,
        metrics=metrics_r,
        rankine_flux=flux_r,
        rankine_ivar=ivar_r,
    )

# ============================
# Select for plot/download
# ============================
if method == "Rankine approach":
    flux_plot_base = results[method]["rankine_flux"]
    ivar_plot_base = results[method]["rankine_ivar"]
else:
    flux_plot_base = flux_s
    ivar_plot_base = ivar_s

cont = results[method]["cont"]
mask_total = results[method]["mask_total"]

cont_safe = np.maximum(cont, 1e-6)
norm_flux = flux_plot_base / cont_safe

norm_err = np.zeros_like(norm_flux, dtype=np.float64)
good_ivar2 = ivar_plot_base > 0
norm_err[good_ivar2] = (1.0 / np.sqrt(ivar_plot_base[good_ivar2])) / cont_safe[good_ivar2]

plot_flux = flux_plot_base.copy()
plot_norm = norm_flux.copy()
if show_masked_as_gaps:
    plot_flux[mask_total] = np.nan
    plot_norm[mask_total] = np.nan

# ============================
# Plotly plot
# ============================
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.4]
)

fig.add_trace(
    go.Scatter(
        x=wave_rest, y=plot_flux,
        mode="lines",
        name="Flux (blue)",
        line=dict(width=1, color="blue"),
        opacity=0.65,
        hovertemplate="λ=%{x:.2f} Å<br>flux=%{y:.4g}<extra></extra>"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=wave_rest, y=cont,
        mode="lines",
        name=f"Continuum ({method}) (red)",
        line=dict(width=3, color="red"),
        hovertemplate="λ=%{x:.2f} Å<br>cont=%{y:.4g}<extra></extra>"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=wave_rest, y=plot_norm,
        mode="lines",
        name="Normalized flux (blue)",
        line=dict(width=1, color="blue"),
        opacity=0.75,
        hovertemplate="λ=%{x:.2f} Å<br>norm=%{y:.4g}<extra></extra>"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=[float(wave_rest.min()), float(wave_rest.max())],
        y=[1.0, 1.0],
        mode="lines",
        line=dict(dash="dash", width=2, color="red"),
        showlegend=False,
        hoverinfo="skip"
    ),
    row=2, col=1
)

# Mask shading
for s, e in mask_to_segments(mask_total):
    fig.add_vrect(
        x0=float(wave_rest[s]), x1=float(wave_rest[e]),
        fillcolor="gray", opacity=0.20, line_width=0
    )

m = results[method]["metrics"]
fig.update_layout(
    height=760,
    title=(
        f"{method} | {pmf} | z={z:.6f} ({z_source}) | "
        f"median_scale={scale:.4g} | masked={int(mask_total.sum())}/{mask_total.size} | "
        f"wRMSE={m['wRMSE']:.4f}"
    ),
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=50),
)

fig.update_yaxes(title_text="Flux (scaled)", row=1, col=1)
fig.update_yaxes(title_text="Normalized", range=[0, 1.6], row=2, col=1)
fig.update_xaxes(title_text="Rest wavelength (Å)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ============================
# Download CSV
# ============================
df_out = pd.DataFrame({
    "rest_wavelength": wave_rest.astype(np.float64),
    "flux": flux_plot_base.astype(np.float64),
    "ivar": ivar_plot_base.astype(np.float64),
    "reconstructed_continuum": cont.astype(np.float64),
    "normalised_flux": norm_flux.astype(np.float64),
    "normalised_error": norm_err.astype(np.float64),
    "mask": mask_total.astype(np.int8),
})

ra = meta.get("ra", None)
dec = meta.get("dec", None)
out_name = f"Norm_{safe_str(ra)}_{safe_str(dec)}_{pmf}_{safe_str(method)}.csv"
csv_bytes = df_out.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download continuum + normalized CSV (selected method)",
    data=csv_bytes,
    file_name=out_name,
    mime="text/csv",
)

# ============================
# Metrics table (default methods + Rankine if available)
# ============================
st.markdown("---")
st.subheader("Fit-quality metrics (computed on mask-free pixels)")

rows = []
for mth in ["PCA", "ICA", "EMPCA", "NMF"]:
    met = results[mth]["metrics"]
    rows.append({
        "Method": mth,
        "N (mask-free)": met["N"],
        "χ²ν (reduced)": met["chi2_nu"],
        "wRMSE": met["wRMSE"],
        "med|z|": met["med_abs_z"],
    })

if "Rankine approach" in results:
    met = results["Rankine approach"]["metrics"]
    rows.append({
        "Method": "Rankine approach",
        "N (mask-free)": met["N"],
        "χ²ν (reduced)": met["chi2_nu"],
        "wRMSE": met["wRMSE"],
        "med|z|": met["med_abs_z"],
    })

df_metrics = pd.DataFrame(rows)

best_idx = df_metrics["wRMSE"].astype(float).idxmin()
best_method = df_metrics.loc[best_idx, "Method"]
best_wrmse = float(df_metrics.loc[best_idx, "wRMSE"])

def highlight_best(row):
    return ["background-color: #FFC9C9"] * len(row) if row.name == best_idx else [""] * len(row)

styled_df = df_metrics.style.apply(highlight_best, axis=1).format({
    "χ²ν (reduced)": "{:.3f}",
    "wRMSE": "{:.4f}",
    "med|z|": "{:.3f}",
})

st.dataframe(styled_df, use_container_width=True)

st.markdown(
    f"""
**Best fit (highlighted above): {best_method}**  
Lowest weighted RMSE = **{best_wrmse:.4f}**  

Selection criterion: minimum **wRMSE**.
"""
)

st.markdown("---")
st.markdown(
    r"""
- **χ²ν (reduced chi-square)**:  
  $$\chi^2_\nu = \frac{\sum (f-c)^2\,\mathrm{ivar}}{\nu},\quad \nu\approx N-1.$$

- **wRMSE (weighted RMSE)**:  
  $$\mathrm{wRMSE}=\sqrt{\frac{\sum \mathrm{ivar}\,(f-c)^2}{\sum \mathrm{ivar}}}.$$

- **med|z| (robust normalized residual)**:  
  $$z=(f-c)\sqrt{\mathrm{ivar}},\qquad \mathrm{med}|z|=\mathrm{median}(|z|).$$
"""
)
