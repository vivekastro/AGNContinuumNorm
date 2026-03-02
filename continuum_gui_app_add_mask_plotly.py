#!/usr/bin/env python3
"""
continuum_gui_app.py  (Streamlit + Plotly)

GUI tool:
- Upload bases NPZ trained on REST-FRAME grid
- Upload a standard SDSS FITS spectrum (observed frame)
- Convert to REST-FRAME using redshift z with precedence:
    (a) user input z_user if > 0
    (b) hdul[2].data['Z'] if available
    (c) header Z/REDSHIFT if available
    (d) fallback z=0
- Interpolate flux/ivar onto model REST-FRAME wavelength grid
- Optionally median-scale BEFORE fitting
- Choose method for plotting & download: PCA / ICA / EMPCA / NMF
- Auto one-sided absorption masking + USER-DEFINED wavelength masks (any number)
- Fit ALL 4 methods, compute metrics on mask-free regions and show a comparison table
- Highlight best method (lowest wRMSE) and print note below
- Interactive Plotly plot (zoomable) with 2 panels (flux+cont, normalized), mask shading
- Download CSV for the selected method with:
    rest_wavelength, flux, ivar, reconstructed_continuum,
    normalised_flux, normalised_error, mask

Run:
  pip install streamlit astropy pandas numpy plotly scikit-learn scipy
  streamlit run continuum_gui_app.py
"""

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.optimize import nnls
except Exception:
    nnls = None


# ----------------------------
# Utils
# ----------------------------
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
    """
    intervals: list of (lo, hi) in same units as wave (REST Å)
    returns boolean mask True inside any interval
    """
    m = np.zeros_like(wave, dtype=bool)
    for lo, hi in intervals:
        if lo is None or hi is None:
            continue
        lo2 = float(min(lo, hi))
        hi2 = float(max(lo, hi))
        m |= (wave >= lo2) & (wave <= hi2)
    return m


def mask_to_segments(mask_bool):
    """
    Convert boolean mask to list of (start_idx, end_idx) contiguous segments.
    """
    m = mask_bool.astype(bool)
    if not np.any(m):
        return []
    idx = np.where(m)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))


# ----------------------------
# FITS reading + z extraction
# ----------------------------
def extract_z_from_hdu2(hdul):
    """
    Redshift is in hdul[2].data['Z'] (your files).
    Returns float or None.
    """
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
    Reader supports:
      A) Table with columns flux/loglam/ivar
      B) 1D primary image with CRVAL1/CD1_1

    Returns:
      wave_obs, flux, ivar, hdr0, z_from_hdu2, meta(dict)
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


# ----------------------------
# Bases + scaling helpers
# ----------------------------
def load_bases_npz(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        d = np.load(io.BytesIO(path_or_bytes), allow_pickle=False)
    else:
        d = np.load(path_or_bytes, allow_pickle=False)
    return {k: d[k] for k in d.files}


def bases_trained_with_robust_scaler(bases):
    flag = bases.get("trained_with_robust_scaler", np.array([0], dtype=np.int8))
    return int(flag[0]) == 1


def apply_training_robust_scaler(X, bases):
    """
    Apply per-wavelength RobustScaler used during training:
        Xs = (X - center) / scale
    """
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


def median_scale_spectrum(flux, trusted):
    if np.any(trusted):
        med = np.median(flux[trusted])
        if np.isfinite(med) and med > 0:
            return float(med)
    return 1.0


# ----------------------------
# Solvers + reconstruction
# ----------------------------
def solve_coeffs_weighted_ls(V, y, w):
    ww = np.sqrt(np.maximum(w, 0.0))
    A = (V.T * ww[:, None])   # LxK
    b = y * ww
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    return a


def solve_coeffs_weighted_nnls(H, y, w):
    """
    Solve a>=0 for (a@H) ~ y, weights w.
    H: KxL
    """
    K, _ = H.shape
    ww = np.sqrt(np.maximum(w, 0.0))
    A = (H.T * ww[:, None])   # LxK
    b = y * ww

    if nnls is not None:
        a, _ = nnls(A, b)
        return a

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
        H = bases["nmf_H"].astype(np.float64)         # K x L
        shift = bases["nmf_shift"].astype(np.float64) # L
        y = (flux_feat + shift)
        a = solve_coeffs_weighted_nnls(H, y, w)
        cont = (a @ H) - shift
        return cont

    raise ValueError(f"Unknown method: {method}")


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(flux_s, cont, ivar_s, trusted_fit, mask_total):
    """
    Computed on mask-free pixels: trusted_fit & ~mask_total & ivar>0.

    chi2_nu: reduced chi-square using ivar weights
    wRMSE  : sqrt( sum(ivar*r^2) / sum(ivar) ) in flux units
    med|z| : median( |(f-c)*sqrt(ivar)| )
    """
    good = trusted_fit & (~mask_total) & np.isfinite(flux_s) & np.isfinite(cont) & (ivar_s > 0)
    n = int(np.sum(good))
    if n < 5:
        return dict(N=n, chi2_nu=np.nan, wRMSE=np.nan, med_abs_z=np.nan)

    r = flux_s[good] - cont[good]
    w = ivar_s[good]

    chi2 = float(np.sum(r * r * w))
    dof = max(n - 1, 1)  # stable proxy (optionally replace with n-K if you store K)
    chi2_nu = chi2 / dof

    wsum = float(np.sum(w))
    wRMSE = np.sqrt(float(np.sum(w * r * r)) / max(wsum, 1e-30))

    z = r * np.sqrt(w)
    med_abs_z = float(np.median(np.abs(z)))

    return dict(N=n, chi2_nu=chi2_nu, wRMSE=wRMSE, med_abs_z=med_abs_z)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="BAL Continuum Normalizer (PCA/ICA/EMPCA/NMF)", layout="wide")
#st.title(" AGN Continuum Reconstruction & Normalization Toolkit ")
st.markdown("### AGN Continuum Reconstruction & Normalization Toolkit")
st.caption("Version 1.3")

with st.expander("About this app"):
    st.markdown(
        """
        **Author:** Vivek M  
        **Institution:** Indian Institute of Astrophysics  

        This tool performs rest-frame continuum reconstruction using:
        - PCA
        - ICA
        - EMPCA
        - NMF

        Includes adaptive masking and user-defined wavelength masking.
        Current version only can handle SDSS spectra.

        📩 For bug reports or feature requests:  
        **vivek.m@iiap.res.in**
        """
    )
st.markdown("---")
with st.sidebar:
    st.header("Inputs")
    bases_file = st.file_uploader("Upload bases NPZ (REST-FRAME grid)", type=["npz"])
    spec_file = st.file_uploader("Upload SDSS FITS spectrum", type=["fits", "fit", "fz"])

    method = st.selectbox("Methodology (for plot & download)", ["PCA", "ICA", "EMPCA", "NMF"], index=2)

    st.subheader("Rest-frame redshift")
    z_user = st.number_input(
        "Redshift z (if >0, overrides file-derived redshift)",
        value=0.0, step=0.001, format="%.6f"
    )

    st.subheader("Scaling")
    use_median_scale = st.checkbox("Per-spectrum median scaling BEFORE fitting (recommended)", value=True)

    st.subheader("Default mask settings")
    k = st.slider("k (sigma): mask if flux < cont - k*sigma", 1.0, 6.0, 2.0, 0.1)
    min_run = st.slider("min_run (pixels)", 1, 20, 3, 1)
    grow = st.slider("grow (pixels)", 0, 15, 1, 1)

    st.subheader("User-defined masks (rest-frame Å)")
    if "user_masks" not in st.session_state:
        st.session_state.user_masks = []  # list of (lo, hi)

    with st.expander("Add / manage masks", expanded=False):
        colA, colB = st.columns(2)
        lo = colA.number_input("Mask λ_low (Å)", value=0.0, step=1.0, format="%.2f")
        hi = colB.number_input("Mask λ_high (Å)", value=0.0, step=1.0, format="%.2f")

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
            st.write("Current masks:")
            st.table(pd.DataFrame(st.session_state.user_masks, columns=["λ_low", "λ_high"]))

    st.subheader("Flux scale option")
    apply_flambda_rest = st.checkbox(
        "Apply strict f_lambda rest-frame scaling: f_rest=(1+z)f_obs, ivar_rest=ivar_obs/(1+z)^2",
        value=False
    )

    run_btn = st.button("Reconstruct + Plot", type="primary")

if bases_file is None or spec_file is None:
    st.info("Upload a **bases NPZ** and a **SDSS FITS** spectrum to begin.")
    st.stop()

bases = load_bases_npz(bases_file.getvalue())

# Load spectrum + z from HDU2 if present
try:
    wave_obs, flux_obs, ivar_obs, hdr0, z_from_hdu2, meta = read_sdss_fits(spec_file.getvalue())
except Exception as e:
    st.error(f"Failed to read FITS: {e}")
    st.stop()

plate = meta.get("plate", None)
mjd = meta.get("mjd", None)
fiberid = meta.get("fiberid", None)
pmf = f"{safe_str(plate)}-{safe_str(mjd)}-{safe_str(fiberid)}"

# ---- z precedence (user > hdu2 > header > 0) ----
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

# Convert observed -> rest wavelength axis
wave_rest_in = wave_obs / (1.0 + z) if (z is not None and z >= 0) else wave_obs

# Optional strict f_lambda rest scaling
flux_in = flux_obs.astype(np.float64).copy()
ivar_in = ivar_obs.astype(np.float64).copy()
if apply_flambda_rest and (z is not None and z >= 0):
    flux_in = flux_in * (1.0 + z)
    ivar_in = ivar_in / ((1.0 + z) ** 2)

# Interpolate onto model REST-FRAME grid
wave_train = bases["wave"].astype(np.float64)

# Overlap sanity check
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
wave_rest = wave_train  # final axis for fit/plot/output

# User-defined masks are in REST-frame wavelength coordinates (wave_rest)
user_mask = intervals_to_mask(wave_rest, st.session_state.user_masks)

# Metadata for filename
ra = meta.get("ra", None)
dec = meta.get("dec", None)

if not run_btn:
    st.stop()

# Trusted pixels: valid & positive ivar, AND not user-masked
trusted_fit = valid_g & (ivar_g > 0) & (~user_mask)

# Per-spectrum median scaling BEFORE fitting
scale = 1.0
if use_median_scale:
    scale = median_scale_spectrum(flux_g, trusted_fit)

flux_s = flux_g / scale
ivar_s = ivar_g * (scale ** 2)

# Precompute sigma once
sigma = np.zeros_like(ivar_s, dtype=np.float64)
good_ivar = ivar_s > 0
sigma[good_ivar] = 1.0 / np.sqrt(ivar_s[good_ivar])

# Fit ALL methods and compute metrics
all_methods = ["PCA", "ICA", "EMPCA", "NMF"]
results = {}  # method -> dict(cont, mask_total, metrics)

for mth in all_methods:
    # Features per method
    if mth in ["PCA", "ICA", "NMF"]:
        Xfeat_2d = flux_s[None, :]
        Xfeat_s_2d, inv_scaler = apply_training_robust_scaler(Xfeat_2d, bases)
        flux_feat = Xfeat_s_2d[0]
    else:
        inv_scaler = None
        flux_feat = flux_s

    # initial fit
    cont0_feat = reconstruct_continuum(mth, bases, flux_feat, ivar_s, trusted_fit)
    if mth in ["PCA", "ICA", "NMF"]:
        cont0 = inverse_training_robust_scaler(cont0_feat[None, :], inv_scaler)[0]
    else:
        cont0 = cont0_feat

    # auto mask from one-sided residuals
    mask_raw = np.zeros_like(flux_s, dtype=bool)
    mask_raw[trusted_fit] = (flux_s[trusted_fit] < (cont0[trusted_fit] - float(k) * sigma[trusted_fit]))
    mask_auto = contiguous_mask(mask_raw, min_run=min_run, grow=grow)

    # total mask includes USER mask too
    mask_total = mask_auto | user_mask

    # refit excluding total mask
    trusted_fit2 = trusted_fit & (~mask_total)
    cont_feat = reconstruct_continuum(mth, bases, flux_feat, ivar_s, trusted_fit2)
    if mth in ["PCA", "ICA", "NMF"]:
        cont = inverse_training_robust_scaler(cont_feat[None, :], inv_scaler)[0]
    else:
        cont = cont_feat

    metrics = compute_metrics(flux_s, cont, ivar_s, trusted_fit, mask_total)
    results[mth] = dict(cont=cont, mask_total=mask_total, metrics=metrics)

# Selected method for plot/download
cont = results[method]["cont"]
mask_total = results[method]["mask_total"]

epsc = 1e-6
cont_safe = np.maximum(cont, epsc)
norm_flux = flux_s / cont_safe
norm_err = np.zeros_like(norm_flux, dtype=np.float64)
norm_err[good_ivar] = (1.0 / np.sqrt(ivar_s[good_ivar])) / cont_safe[good_ivar]

# ----------------------------
# Interactive Plotly plot (zoomable)
# ----------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.4]
)

fig.add_trace(
    go.Scatter(
        x=wave_rest,
        y=flux_s,
        mode="lines",
        name="Flux (scaled)",
        line=dict(width=1,color='blue'),
        opacity=0.6,
        hovertemplate="λ=%{x:.2f} Å<br>flux=%{y:.4g}<extra></extra>"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=wave_rest,
        y=cont,
        mode="lines",
        name=f"Continuum ({method})",
        line=dict(width=3,color='red'),
        hovertemplate="λ=%{x:.2f} Å<br>cont=%{y:.4g}<extra></extra>"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=wave_rest,
        y=norm_flux,
        mode="lines",
        name="Normalized flux",
        line=dict(width=1,color='blue'),
        opacity=0.6,
        hovertemplate="λ=%{x:.2f} Å<br>norm=%{y:.4g}<extra></extra>"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=[float(wave_rest.min()), float(wave_rest.max())],
        y=[1.0, 1.0],
        mode="lines",
        line=dict(dash="dash",width=2,color='red'),
        showlegend=False,
        hoverinfo="skip"
    ),
    row=2, col=1
)

# Mask shading
segments = mask_to_segments(mask_total)
for s, e in segments:
    fig.add_vrect(
        x0=float(wave_rest[s]),
        x1=float(wave_rest[e]),
        fillcolor="gray",
        opacity=0.2,
        line_width=0
    )

fig.update_layout(
    height=750,
    title=(
        f"{method} | {pmf} | z={z:.6f} ({z_source}) | "
        f"median_scale={scale:.4g} | masked={int(mask_total.sum())}/{mask_total.size}"
    ),
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=50),
)

fig.update_yaxes(title_text="Flux (scaled)", row=1, col=1)
fig.update_yaxes(title_text="Normalized", range=[0, 1.6], row=2, col=1)
fig.update_xaxes(title_text="Rest wavelength (Å)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Output CSV for selected method
# ----------------------------
df_out = pd.DataFrame({
    "rest_wavelength": wave_rest.astype(np.float64),
    "flux": flux_s.astype(np.float64),
    "ivar": ivar_s.astype(np.float64),
    "reconstructed_continuum": cont.astype(np.float64),
    "normalised_flux": norm_flux.astype(np.float64),
    "normalised_error": norm_err.astype(np.float64),
    "mask": mask_total.astype(np.int8),
})

out_name = f"Norm_{safe_str(ra)}_{safe_str(dec)}_{pmf}_{method}.csv"
csv_bytes = df_out.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download continuum + normalized CSV (selected method)",
    data=csv_bytes,
    file_name=out_name,
    mime="text/csv",
)
# ----------------------------
# Metrics table + highlight best
# ----------------------------
rows = []
for mth in all_methods:
    met = results[mth]["metrics"]
    rows.append({
        "Method": mth,
        "N (mask-free)": met["N"],
        "χ²ν (reduced)": met["chi2_nu"],
        "wRMSE": met["wRMSE"],
        "med|z|": met["med_abs_z"],
    })
df_metrics = pd.DataFrame(rows)
st.markdown("---")
st.subheader("Fit-quality metrics (computed on mask-free regions)")

best_idx = df_metrics["wRMSE"].astype(float).idxmin()
best_method = df_metrics.loc[best_idx, "Method"]
best_wrmse = float(df_metrics.loc[best_idx, "wRMSE"])

st.markdown(
    f"""
**Best fit (highlighted above): {best_method}**  
Lowest weighted RMSE = **{best_wrmse:.4f}**  

Selection criterion: minimum **wRMSE** (weighted root-mean-square residual in flux units).
"""
)

def highlight_best(row):
    if row.name == best_idx:
        return ["background-color: #FFC9C9"] * len(row)
    return [""] * len(row)

styled_df = df_metrics.style.apply(highlight_best, axis=1).format({
    "χ²ν (reduced)": "{:.3f}",
    "wRMSE": "{:.4f}",
    "med|z|": "{:.3f}",
})

st.dataframe(styled_df, use_container_width=True)
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



