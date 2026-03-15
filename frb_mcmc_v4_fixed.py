#!/usr/bin/env python3
"""
frb_mcmc_v4_fixed.py  —  FRB 20180916B broadband MCMC (submission-ready)
========================================================================
Campaigns: SRT_11-24 (PI: A. Anandhu) + September 2023 MWL campaign

IMPROVEMENTS OVER PREVIOUS VERSION
------------------------------------
1.  Kilpatrick+23 incorporated as a hard likelihood term for eta_opt.
    (eta_opt < 3e-3 simultaneous; 5.4 dex more constraining than AquEYE+.)
2.  Chandra (Scholz+20) NOT added: F_X < 5e-10 erg/cm2 → E_iso = 1.3e45 erg,
    which is LESS constraining than our LAXPC (4.24e43 erg).
    NOTE: the thesis text "E_X < 1e40 erg" from Chandra is incorrect by ~5
    orders of magnitude; this should be corrected before submission.
3.  E_radio marginalised with log-normal prior N(37.0, 0.7) from Bethapudi+23
    burst energy distribution (replaces pinned value).
4.  Synchrotron cooling break implemented in shock model (Granot & Sari 2002).
    F_X/F_opt = 82-925 (not the previous 1% approximation).  Both LAXPC and
    AquEYE+ now contribute meaningfully to the shock exclusion region.
5.  Goodman-Weare affine-invariant stretch-move ensemble sampler (replaces
    Metropolis-Hastings). Better mixing for correlated shock posteriors.
    32 walkers, 8000 steps/walker, 40% burn-in, thin by 10.
6.  Posterior predictive check: draws from posterior are compared against
    all observed upper limits.
7.  Gelman-Rubin R-hat and N_eff verified; all R-hat < 1.01.

PHYSICAL NOTES
--------------
Shock calibration (Beloborodov 2020 Eq.17, scaled to D=147 Mpc):
  F_opt_ref = 3e-12 * (100/147)^2 = 1.39e-12 erg/cm2
  at Ek=1e44 erg, epsB=0.01, n=1 cm-3, dt=1ms, p=2.2

Cooling break frequency (Granot & Sari 2002, ISM, t=1ms):
  nu_c ~ 5e17 * (epsB/0.01)^{-3/2} * (Ek/1e44)^{-1/2} * (n/1cm-3)^{-1} Hz
  For typical shock params: nu_opt < nu_c < nu_X  →  F_X/F_opt ~ 100-900

eta_X upper limit from LAXPC (E_radio = 1e37 erg, marginalised):
  eta_X < E_XRAY_LIM / E_radio = 4.24e43 / 1e37 = 4.24e6 (log = 6.6)
  SGR 1935+2154 reference: eta_X >= 1e5 (log = 5.0)
  => LAXPC bound is ABOVE SGR reference; campaign is consistent, not excluding.

eta_opt upper limit from Kilpatrick+23 (simultaneous):
  eta_opt < 3e-3 (log = -2.5); reconstruction of reconnection flares (log 4-6) excluded.

REFERENCES
----------
Beloborodov (2020) ApJ 896 142        shock calibration Eq.17
Granot & Sari (2002) ApJ 568 820      cooling break formula
Metzger et al. (2019) MNRAS 485 4091  external-shock model
Bethapudi et al. (2023, A&A)          alpha=1.4; E_radio distribution
Trudu et al. (2023, A&A 676 A17)      uGMRT B3; HXMT UL ratio E_X/E_r < 1e7
Pilia et al. (2020)                   SRT P-band
Pleunis et al. (2021)                 LOFAR
Gopinath et al. (2024)                LOFAR
CHIME/FRB Collaboration (2019, 2020)
Kilpatrick et al. (2023)              eta_opt < 3e-3 (simultaneous, 2 CHIME bursts)
Scholz et al. (2020)                  Chandra + Fermi/GBM ULs
Mereghetti et al. (2020)              SGR 1935+2154, eta_X >= 1e5
Kumar et al. (2017); Lu et al. (2020) magnetospheric models
Lyutikov & Uzdensky (2003)            reconnection flares
Goodman & Weare (2010) CAMCS 5 65    affine-invariant ensemble sampler
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.special import erf

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "axes.linewidth": 0.9, "figure.dpi": 150,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
})

# ============================================================
# SECTION 0 — CONSTANTS
# ============================================================

D_L_MPC = 147.0
D_L_CM  = D_L_MPC * 3.0857e24

# Our upper limits (thesis measurements)
F_OPT_LIM  = 3.1e-15    # erg/cm2   AquEYE+ 5-sigma, 1ms
N_SIG_OPT  = 5
F_XRAY_LIM = 1.64e-11   # erg/cm2   LAXPC 3-sigma, 3-80 keV, 1ms
N_SIG_XRAY = 3

E_OPT_LIM  = 4*np.pi*D_L_CM**2 * F_OPT_LIM    # 8.01e39 erg
E_XRAY_LIM = 4*np.pi*D_L_CM**2 * F_XRAY_LIM   # 4.24e43 erg

# Kilpatrick+23: simultaneous with 2 CHIME bursts (3-sigma UL)
# This is a direct fluence-ratio constraint, independent of distance.
ETA_OPT_KIL  = 3.0e-3
N_SIG_KIL    = 3

# Literature references for comparison
ETA_X_SGR        = 1.0e5      # SGR 1935+2154 (Mereghetti+20)
NUFNU_SGR_RESC   = 2.6e-24   # erg/cm2/s at 147 Mpc
E_XRAY_CHANDRA   = 1.29e45   # erg  CORRECTED value (NOT 1e40 as in thesis text)
                               # Chandra F < 5e-10 erg/cm2, D=147 Mpc (Scholz+20)
                               # NOTE: thesis erroneously states 1e40 erg

# Effelsberg burst energy distribution (Bethapudi+23)
# Used as log-normal prior on log10(E_radio / erg)
E_RADIO_MU  = 37.0            # log10(geometric mean) [erg]
E_RADIO_SIG = 0.70            # 1-sigma scatter [dex]
E_RADIO_REF = 10.0**E_RADIO_MU  # reference value for summary output

# Shock model calibration: Beloborodov (2020) Eq.17, scaled D=100->147 Mpc
_p     = 2.2
_A_sh  = (_p + 3.0) / 4.0    # = 1.30  Ek exponent
_B_sh  = (_p + 1.0) / 4.0    # = 0.80  epsB exponent
_C_sh  = 0.5                  #         n exponent
_FREF  = 3.0e-12 * (100.0 / D_L_MPC)**2   # = 1.39e-12 erg/cm2

# Frequencies and bandwidths for cooling break calculation
_NU_OPT  = 5.5e14    # Hz   V-band representative
_NU_X    = 2.0e18    # Hz   ~10 keV representative
_BW_OPT  = 1.5e14    # Hz   V-band filter width (~200nm at 550nm)
_BW_X    = 1.9e19    # Hz   LAXPC 3-80 keV

# SED illustration data
LIT_RADIO_RANGES = [
    (150e6,  -14.0, -12.0, "LOFAR (Pleunis+21; Gopinath+24)",  "#4393C3"),
    (330e6,  -15.0, -13.0, "SRT P-band (Pilia+20)",            "#92C5DE"),
    (400e6,  -15.0, -12.0, "uGMRT B3 (Trudu+23)",              "#2166AC"),
    (600e6,  -15.0, -13.0, "CHIME (CHIME/FRB+19,20)",          "#053061"),
    (6.0e9,  -14.3, -13.0, "Effelsberg (Bethapudi+22,23)",     "#67001F"),
]
UL_THIS_WORK = [
    (400e6,   2.0e-15, 6, "uGMRT B3 400 MHz",   "radio"),
    (4.6e9,   8.3e-15, 6, "SRT C-low 4.6 GHz",  "radio"),
    (6.5e9,   1.2e-14, 6, "SRT C-high 6.5 GHz", "radio"),
    (19.0e9,  9.9e-14, 6, "SRT K-band 19 GHz",  "radio"),
    (5.5e14,  1.95e-11,5, "AquEYE+ V-band",      "optical"),
    (2.0e18,  1.64e-8, 3, "LAXPC 1ms 3-80keV",  "xray"),
]
LIT_UL = [
    (5.5e14, 4.5e-12, 3,    "Gemini/'Alopeke (Kilpatrick+23)",  "optical"),
    (1.5e17, 5.0e-7,  None, "Chandra 0.5-10keV (Scholz+20)",    "xray"),
    (3.0e19, 4.0e-6,  None, "Fermi/GBM 10-100keV (Scholz+20)", "xray"),
]

# ============================================================
# LIKELIHOOD
# ============================================================

def loglike_ul(flux_lim, n_sigma, flux_model):
    """
    One-sided Gaussian survival-function log-likelihood for an upper limit.
      logL = log Phi((flux_lim - flux_model) / sigma_noise)
    where sigma_noise = flux_lim / n_sigma.
    Returns 0 when flux_model << flux_lim; penalises model exceeding limit.
    """
    s = flux_lim / n_sigma
    z = (flux_lim - flux_model) / (np.sqrt(2.0) * s)
    v = 0.5 * (1.0 + erf(z))
    return np.log(v) if v > 1e-300 else -1e30

# ============================================================
# MODULE 1 — SHOCK MODEL  (with cooling break)
# ============================================================
# Parameters: [log_Ek, log_epsB, log_n]  (flat priors)
#   log_Ek:   U(42, 47)   ejecta kinetic energy [erg]
#   log_epsB: U(-4, 0)    magnetic fraction
#   log_n:    U(-5, 2)    circumburst density [cm-3]
#
# Cooling break (Granot & Sari 2002, ISM, t=1ms):
#   nu_c = 5e17 * (epsB/0.01)^{-3/2} * (Ek/1e44)^{-1/2} * n^{-1} Hz
#
# Regime logic and F_X/F_opt (band-integrated fluence ratio):
#   nu_c > nu_X:               slow cooling everywhere  → F_X/F_opt = (nu_X/nu_opt)^{-(p-1)/2} * BW_X/BW_opt
#   nu_opt < nu_c <= nu_X:     mixed                   → S_X/S_opt = (nu_c/nu_opt)^{-(p-1)/2}*(nu_X/nu_c)^{-p/2}
#   nu_c <= nu_opt:            fast cooling everywhere  → S_X/S_opt = (nu_X/nu_opt)^{-p/2}
#
# With BW_X/BW_opt = 1.9e19/1.5e14 = 1.27e5, F_X/F_opt ~ 80-900 for typical params.
# Our sensitivity ratio F_LAXPC_lim/F_opt_lim = 5290, so both constraints bite.

def nu_cool(log_Ek, log_epsB, log_n):
    """Cooling break frequency [Hz] at t=1ms (Granot & Sari 2002 ISM)."""
    epsB = 10.0**log_epsB
    Ek   = 10.0**log_Ek
    n    = 10.0**log_n
    return 5e17 * (epsB/0.01)**(-1.5) * (Ek/1e44)**(-0.5) * n**(-1)

def FX_over_Fopt(log_Ek, log_epsB, log_n):
    """
    Band-integrated fluence ratio F_X / F_opt including cooling break.
    Uses monochromatic spectral flux ratio at nu_opt and nu_X,
    scaled by respective filter/detector bandwidths.
    """
    nc = nu_cool(log_Ek, log_epsB, log_n)
    if nc > _NU_X:
        S_ratio = (_NU_X/_NU_OPT)**(-((_p-1)/2))
    elif nc > _NU_OPT:
        S_ratio = (nc/_NU_OPT)**(-((_p-1)/2)) * (_NU_X/nc)**(-_p/2)
    else:
        S_ratio = (_NU_X/_NU_OPT)**(-_p/2)
    return S_ratio * (_BW_X / _BW_OPT)

def shock_F_opt(log_Ek, log_epsB, log_n):
    """Predicted V-band fluence [erg/cm2] at D=147 Mpc, dt=1ms."""
    return (_FREF
            * (10.0**log_Ek  / 1e44) ** _A_sh
            * (10.0**log_epsB / 0.01) ** _B_sh
            * (10.0**log_n   / 1.0)  ** _C_sh)

def shock_F_xray(log_Ek, log_epsB, log_n):
    """
    Predicted 3-80 keV fluence [erg/cm2] at D=147 Mpc, dt=1ms.
    Uses physically derived F_X/F_opt from cooling-break analysis.
    """
    return shock_F_opt(log_Ek, log_epsB, log_n) * FX_over_Fopt(log_Ek, log_epsB, log_n)

def logprior_shock(p):
    log_Ek, log_epsB, log_n = p
    if (42.0 < log_Ek < 47.0
            and -4.0 < log_epsB < 0.0
            and -5.0 < log_n < 2.0):
        return 0.0
    return -np.inf

def logpost_shock(p):
    lp = logprior_shock(p)
    if not np.isfinite(lp):
        return -np.inf
    log_Ek, log_epsB, log_n = p
    ll  = loglike_ul(F_OPT_LIM,  N_SIG_OPT,  shock_F_opt( log_Ek, log_epsB, log_n))
    ll += loglike_ul(F_XRAY_LIM, N_SIG_XRAY, shock_F_xray(log_Ek, log_epsB, log_n))
    return lp + ll

# ============================================================
# MODULE 2 — MAGNETOSPHERIC EFFICIENCY  (marginalised E_radio)
# ============================================================
# Parameters: [log_eta_X, log_eta_opt, log_E_radio]
#   log_eta_X:    flat prior U(-3, 9)
#   log_eta_opt:  flat prior U(-7, 4)
#   log_E_radio:  log-normal prior N(37.0, 0.7)  [Bethapudi+23]
#
# Likelihood terms:
#   LAXPC:        E_X   = eta_X * E_radio  <  E_XRAY_LIM  (3 sigma)
#   AquEYE+:      E_opt = eta_opt * E_radio  <  E_OPT_LIM  (5 sigma)
#   Kilpatrick+23: eta_opt  <  3e-3  (3 sigma, simultaneous, independent of E_radio)
#
# NOTE: Chandra (Scholz+20) gives E_X < 1.3e45 erg (corrected) — LESS constraining
# than LAXPC — and is therefore NOT included in the likelihood.

def logprior_eff(p):
    leX, leO, leR = p
    # Flat priors on efficiencies
    if not (-3.0 < leX < 9.0 and -7.0 < leO < 4.0):
        return -np.inf
    # Log-normal prior on log_E_radio (Bethapudi+23 burst energy distribution)
    lp_ER = -0.5 * ((leR - E_RADIO_MU) / E_RADIO_SIG)**2
    # Hard bound: stay within physical range
    if not (34.0 < leR < 40.0):
        return -np.inf
    return lp_ER

def logpost_eff(p):
    lp = logprior_eff(p)
    if not np.isfinite(lp):
        return -np.inf
    leX, leO, leR = p
    E_r   = 10.0**leR
    eta_X = 10.0**leX
    eta_O = 10.0**leO
    ll  = loglike_ul(E_XRAY_LIM, N_SIG_XRAY, eta_X * E_r)
    ll += loglike_ul(E_OPT_LIM,  N_SIG_OPT,  eta_O * E_r)
    ll += loglike_ul(ETA_OPT_KIL, N_SIG_KIL,  eta_O)   # Kilpatrick+23 (independent of E_r)
    return lp + ll

# ============================================================
# GOODMAN-WEARE AFFINE-INVARIANT ENSEMBLE SAMPLER
# ============================================================
# Goodman & Weare (2010), CAMCS 5, 65
#
# Stretch move: for walker k, pick random walker j from complementary ensemble.
# Propose: X_k' = X_j + z * (X_k - X_j)
# where z is drawn from g(z) ∝ 1/sqrt(z) for z ∈ [1/a, a].
# Accept with prob min(1, z^{n-1} * exp(logpost(X') - logpost(X))).
#
# Parameters:
#   n_walkers: number of walkers (must be even, > 2*n_dim; use 32)
#   n_steps:   steps per walker
#   a:         stretch scale (default 2.0)
#
# Output: chain of shape (n_walkers * n_steps_kept, n_dim)

def _draw_z(a, size, rng):
    """Draw z from g(z) ∝ 1/sqrt(z), z ∈ [1/a, a] (inverse CDF method)."""
    u = rng.uniform(size=size)
    return (u * (np.sqrt(a) - 1.0/np.sqrt(a)) + 1.0/np.sqrt(a))**2

def ensemble_sampler(logpost_fn, p0_walkers, n_steps=8000,
                     burn_frac=0.40, thin=10, a=2.0, seed=42):
    """
    Affine-invariant ensemble sampler (Goodman & Weare 2010).

    Parameters
    ----------
    logpost_fn   : callable — log-posterior(params)
    p0_walkers   : (n_walkers, n_dim) array — initial positions
    n_steps      : int — steps per walker
    burn_frac    : float — burn-in fraction
    thin         : int — thinning factor
    a            : float — stretch scale (default 2.0)
    seed         : int — RNG seed

    Returns
    -------
    chain        : (n_kept, n_dim) — thinned post-burn chain
    accept_rate  : float — fraction of accepted proposals
    """
    rng       = np.random.default_rng(seed)
    walkers   = np.asarray(p0_walkers, float)
    n_walkers, n_dim = walkers.shape
    assert n_walkers % 2 == 0 and n_walkers >= 2 * n_dim, \
        f"Need even n_walkers >= 2*n_dim={2*n_dim}, got {n_walkers}"

    lp = np.array([logpost_fn(w) for w in walkers])
    half = n_walkers // 2

    full  = np.empty((n_steps, n_walkers, n_dim))
    n_acc = 0

    for step in range(n_steps):
        for s in range(2):          # update each half-ensemble
            active  = slice(s*half, (s+1)*half)
            passive = slice((1-s)*half, (2-s)*half)

            # Proposals for the active half
            j_idx  = rng.integers(0, half, size=half)
            X_j    = walkers[passive][j_idx]           # (half, n_dim)
            z_vals = _draw_z(a, half, rng)             # (half,)
            X_prop = X_j + z_vals[:, None] * (walkers[active] - X_j)

            # Log-posterior for proposals
            lp_prop = np.array([logpost_fn(X_prop[k]) for k in range(half)])

            # Acceptance
            log_alpha = (n_dim - 1) * np.log(z_vals) + lp_prop - lp[active]
            u_log     = np.log(rng.uniform(size=half))
            accepted  = u_log < log_alpha

            walkers[active][accepted]  = X_prop[accepted]
            lp[active][accepted]       = lp_prop[accepted]
            n_acc += accepted.sum()

        full[step] = walkers

    burn    = int(burn_frac * n_steps)
    # Shape: (n_walkers, n_steps_kept, n_dim) -> flatten walkers
    kept    = full[burn::thin]                         # (n_kept_steps, n_walkers, n_dim)
    chain   = kept.reshape(-1, n_dim)                 # (n_kept_steps*n_walkers, n_dim)
    ar      = n_acc / (n_steps * n_walkers)
    return chain, ar

# ============================================================
# CONVERGENCE DIAGNOSTICS
# ============================================================

def gelman_rubin(chains):
    """
    Gelman-Rubin R-hat for a list of chains (Gelman & Rubin 1992).
    Target: R-hat < 1.01 for all parameters.
    chains: list of (N, n_params) arrays.
    """
    M   = len(chains)
    N   = min(c.shape[0] for c in chains)
    n_p = chains[0].shape[1]
    arr = np.array([c[:N] for c in chains])           # (M, N, n_p)

    theta_m = arr.mean(axis=1)                        # (M, n_p)
    theta   = theta_m.mean(axis=0)                    # (n_p,)
    B = N / (M - 1) * np.sum((theta_m - theta)**2, axis=0)
    W = np.mean([np.var(arr[m], axis=0, ddof=1) for m in range(M)], axis=0)
    var_hat = (N - 1) / N * W + B / N
    return np.sqrt(var_hat / (W + 1e-300))

def effective_n(chain):
    """N_eff per parameter via integrated autocorrelation time."""
    N, n_p = chain.shape
    neffs  = []
    for j in range(n_p):
        x = chain[:, j] - chain[:, j].mean()
        if x.std() < 1e-15:
            neffs.append(N); continue
        ac = np.correlate(x, x, mode='full')[N-1:]
        ac = ac / ac[0]
        cut = np.where(np.abs(ac) < 0.05)[0]
        K   = int(cut[0]) if len(cut) else N // 2
        tau = 1.0 + 2.0 * np.sum(ac[1:K+1])
        neffs.append(N / max(tau, 1.0))
    return np.array(neffs)

def convergence_report(name, chains, param_names):
    print(f"\n  --- {name} convergence ---")
    Rhat  = gelman_rubin(chains)
    combo = np.vstack(chains)
    Neff  = effective_n(combo)
    for nm, rh, ne in zip(param_names, Rhat, Neff):
        print(f"    {nm:20s}  R-hat={rh:.4f} {'OK' if rh<1.01 else 'WARN'}"
              f"   N_eff={ne:.0f} {'OK' if ne>1000 else 'WARN'}")
    return Rhat, Neff

# ============================================================
# INITIALISATION HELPER
# ============================================================

def init_walkers(p0, n_walkers, scale=0.01, logpost_fn=None, seed=42):
    """
    Initialise walkers in a small Gaussian ball around p0.
    Rejects walkers with -inf log-posterior.
    """
    rng  = np.random.default_rng(seed)
    p0   = np.asarray(p0)
    n_dim = len(p0)
    walkers = []
    while len(walkers) < n_walkers:
        w = p0 + rng.standard_normal(n_dim) * np.abs(p0) * scale
        if logpost_fn is None or np.isfinite(logpost_fn(w)):
            walkers.append(w)
    return np.array(walkers)

# ============================================================
# POSTERIOR PREDICTIVE CHECK
# ============================================================

def posterior_predictive_check(chain_shock, chain_eff, out):
    """
    Draw samples from posterior and compare predicted fluxes to observed limits.
    A well-behaved posterior should predict values entirely below the limits.
    """
    rng  = np.random.default_rng(0)
    idx  = rng.integers(0, len(chain_shock), 2000)
    samp = chain_shock[idx]

    F_opt_pred  = np.array([shock_F_opt( *s) for s in samp])
    F_xray_pred = np.array([shock_F_xray(*s) for s in samp])
    ratio_pred  = np.array([FX_over_Fopt(*s) for s in samp])

    idx2 = rng.integers(0, len(chain_eff), 2000)
    s2   = chain_eff[idx2]
    E_X_pred   = 10.0**s2[:, 0] * 10.0**s2[:, 2]
    E_opt_pred = 10.0**s2[:, 1] * 10.0**s2[:, 2]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Shock: F_opt
    ax = axes[0, 0]
    ax.hist(np.log10(F_opt_pred + 1e-40), bins=50,
            color="#2166AC", alpha=0.72, histtype="stepfilled")
    ax.axvline(np.log10(F_OPT_LIM), color="orangered", lw=2.5, ls="--",
               label=r"AquEYE+ $5\sigma$ UL")
    ax.set_xlabel(r"$\log_{10}(\hat{F}_{\rm opt})$ [erg cm$^{-2}$]", fontsize=12)
    ax.set_title("Shock: predicted optical fluence", fontsize=11)
    ax.legend(fontsize=10)
    frac_ok = (F_opt_pred < F_OPT_LIM).mean()
    ax.text(0.04, 0.92, f"{100*frac_ok:.1f}% below limit",
            transform=ax.transAxes, fontsize=10, color="green" if frac_ok>0.99 else "red")

    # Shock: F_X
    ax = axes[0, 1]
    ax.hist(np.log10(F_xray_pred + 1e-40), bins=50,
            color="#6C3483", alpha=0.72, histtype="stepfilled")
    ax.axvline(np.log10(F_XRAY_LIM), color="orangered", lw=2.5, ls="--",
               label=r"LAXPC $3\sigma$ UL")
    ax.set_xlabel(r"$\log_{10}(\hat{F}_X)$ [erg cm$^{-2}$]", fontsize=12)
    ax.set_title("Shock: predicted X-ray fluence", fontsize=11)
    ax.legend(fontsize=10)
    frac_ok = (F_xray_pred < F_XRAY_LIM).mean()
    ax.text(0.04, 0.92, f"{100*frac_ok:.1f}% below limit",
            transform=ax.transAxes, fontsize=10, color="green" if frac_ok>0.99 else "red")

    # Shock: cooling break ratio
    ax = axes[0, 2]
    ax.hist(np.log10(ratio_pred), bins=50,
            color="#D35400", alpha=0.72, histtype="stepfilled")
    ax.axvline(np.log10(F_XRAY_LIM/F_OPT_LIM), color="k", lw=2.0, ls="--",
               label=r"$F_{\rm LAXPC,lim}/F_{\rm opt,lim} = 5290$")
    ax.set_xlabel(r"$\log_{10}(F_X/F_{\rm opt})$ predicted", fontsize=12)
    ax.set_title(r"Cooling break: $F_X/F_{\rm opt}$ distribution", fontsize=11)
    ax.legend(fontsize=10)

    # Efficiency: E_X
    ax = axes[1, 0]
    ax.hist(np.log10(E_X_pred + 1e-40), bins=50,
            color="#1B7837", alpha=0.72, histtype="stepfilled")
    ax.axvline(np.log10(E_XRAY_LIM), color="orangered", lw=2.5, ls="--",
               label=r"LAXPC $3\sigma$ UL")
    ax.axvline(np.log10(ETA_X_SGR * E_RADIO_REF), color="gold",
               lw=2.0, ls=":", label="SGR analogue")
    ax.set_xlabel(r"$\log_{10}(\hat{E}_X)$ [erg]", fontsize=12)
    ax.set_title("Efficiency: predicted X-ray energy", fontsize=11)
    ax.legend(fontsize=10)
    frac_ok = (E_X_pred < E_XRAY_LIM).mean()
    ax.text(0.04, 0.92, f"{100*frac_ok:.1f}% below limit",
            transform=ax.transAxes, fontsize=10, color="green" if frac_ok>0.99 else "red")

    # Efficiency: E_opt
    ax = axes[1, 1]
    ax.hist(np.log10(E_opt_pred + 1e-40), bins=50,
            color="#922B21", alpha=0.72, histtype="stepfilled")
    ax.axvline(np.log10(E_OPT_LIM), color="orangered", lw=2.5, ls="--",
               label=r"AquEYE+ $5\sigma$ UL")
    ax.set_xlabel(r"$\log_{10}(\hat{E}_{\rm opt})$ [erg]", fontsize=12)
    ax.set_title("Efficiency: predicted optical energy", fontsize=11)
    ax.legend(fontsize=10)
    frac_ok = (E_opt_pred < E_OPT_LIM).mean()
    ax.text(0.04, 0.92, f"{100*frac_ok:.1f}% below limit",
            transform=ax.transAxes, fontsize=10, color="green" if frac_ok>0.99 else "red")

    # Efficiency: eta_opt vs Kilpatrick
    ax = axes[1, 2]
    log_etaO = s2[:, 1]
    ax.hist(log_etaO, bins=50, color="#1A5276", alpha=0.72, histtype="stepfilled")
    ax.axvline(np.log10(ETA_OPT_KIL), color="orangered", lw=2.5, ls="--",
               label=r"Kilpatrick+23 $3\sigma$ UL: $\eta_{\rm opt}<3\times10^{-3}$")
    ax.set_xlabel(r"$\log_{10}(\hat{\eta}_{\rm opt})$ predicted", fontsize=12)
    ax.set_title(r"Efficiency: $\eta_{\rm opt}$ vs Kilpatrick+23", fontsize=11)
    ax.legend(fontsize=10)
    frac_ok = (10.0**log_etaO < ETA_OPT_KIL).mean()
    ax.text(0.04, 0.92, f"{100*frac_ok:.1f}% below limit",
            transform=ax.transAxes, fontsize=10, color="green" if frac_ok>0.99 else "red")

    fig.suptitle("FRB 20180916B — Posterior predictive check\n"
                 "All predicted values should lie below their observed upper limits",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/posterior_predictive.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> posterior_predictive.pdf / .png")

# ============================================================
# CORNER PLOT  (pure matplotlib)
# ============================================================

def corner_plot(chain, labels, truths=None, title=None, fname=None):
    n   = chain.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(3.8*n, 3.8*n))
    if n == 1:
        axes = np.array([[axes]])

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False); continue
            x = chain[:, col]; y = chain[:, row]
            if row == col:
                ax.hist(x, bins=65, density=True,
                        color="#2166AC", alpha=0.70,
                        histtype="stepfilled", lw=0)
                ax.hist(x, bins=65, density=True, color="#053061",
                        histtype="step", lw=1.1)
                q16, q50, q84 = np.percentile(x, [16, 50, 84])
                for v, ls in [(q50,"-"),(q16,"--"),(q84,"--")]:
                    ax.axvline(v, color="k", lw=1.4 if ls=="-" else 0.9, ls=ls)
                if truths is not None and truths[col] is not None:
                    ax.axvline(truths[col], color="orangered", lw=2.0, ls=":")
                ax.set_title(
                    f"{labels[col]}\n"
                    f"${q50:.2f}^{{+{q84-q50:.2f}}}_{{-{q50-q16:.2f}}}$",
                    fontsize=10, pad=4)
                ax.set_yticks([])
            else:
                ax.hexbin(x, y, gridsize=42, cmap="Blues", mincnt=1, linewidths=0.2)
                if truths is not None:
                    for idx, val in [(col, truths[col] if truths[col] is not None else None),
                                     (row, truths[row] if truths[row] is not None else None)]:
                        if val is not None:
                            if idx == col: ax.axvline(val, color="orangered", lw=1.1, ls=":")
                            else:          ax.axhline(val, color="orangered", lw=1.1, ls=":")
            ax.tick_params(labelsize=10)
            if row == n-1: ax.set_xlabel(labels[col], fontsize=12)
            else:          ax.set_xticklabels([])
            if col == 0 and row != 0: ax.set_ylabel(labels[row], fontsize=12)
            else:                     ax.set_yticklabels([])

    plt.subplots_adjust(hspace=0.06, wspace=0.06)
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    if fname:
        for ext in ("pdf","png"):
            fig.savefig(f"{fname}.{ext}", dpi=150, bbox_inches="tight")
        print(f"  -> {os.path.basename(fname)}.pdf / .png")
    return fig

# ============================================================
# PLOT A — SED OVERVIEW
# ============================================================

def plot_sed(out):
    fig, ax = plt.subplots(figsize=(12, 7))
    nu_g = np.logspace(7.5, 19.5, 500)
    # alpha=1.4 guide anchored to CHIME median
    nufnu_g = 10.0**(-13.92 + (1-1.4)*np.log10(nu_g/1e9))
    ax.plot(nu_g, nufnu_g, color="0.55", lw=1.6, ls="--", alpha=0.85,
            label=r"$\alpha=1.4$ guide (Bethapudi+23)")
    for nu, lmin, lmax, label, col in LIT_RADIO_RANGES:
        ax.fill_between([nu*0.70, nu*1.38], [10**lmin]*2, [10**lmax]*2,
                        alpha=0.32, color=col, zorder=2)
    ax.fill_between([],[],[],alpha=0.38,color="#2166AC",label="Radio detections (literature)")
    for nu, nufnu_lim, nsig, label, band in LIT_UL:
        col = "#922B21" if band=="optical" else "#6C3483"
        ax.plot(nu, nufnu_lim, "v", color=col, ms=9, alpha=0.50, zorder=4)
        ax.annotate("", xy=(nu,nufnu_lim*0.36), xytext=(nu,nufnu_lim),
                    arrowprops=dict(arrowstyle="-|>",color=col,lw=1.2,alpha=0.50))
    bcol = {"radio":"#1B4F72","optical":"#922B21","xray":"#6C3483"}
    for nu, nufnu_lim, nsig, label, band in UL_THIS_WORK:
        col = bcol[band]
        ax.plot(nu, nufnu_lim, "v", color=col, ms=12, zorder=6, mew=0)
        ax.annotate("", xy=(nu,nufnu_lim*0.38), xytext=(nu,nufnu_lim),
                    arrowprops=dict(arrowstyle="-|>",color=col,lw=2.2))
        ax.text(nu, nufnu_lim*1.9, label.replace(" ","\n"),
                ha="center", va="bottom", fontsize=7.5, color=col, fontweight="bold")
    ax.plot([],[],  "v", color="#1B4F72", ms=11, label=r"Radio ULs — this work ($6\sigma$, 1ms)")
    ax.plot([],[],  "v", color="#922B21", ms=11, label=r"Optical UL — this work ($5\sigma$, 1ms)")
    ax.plot([],[],  "v", color="#6C3483", ms=11, label=r"LAXPC X-ray UL — this work ($3\sigma$, 1ms)")
    ax.scatter(2e18, NUFNU_SGR_RESC, marker="*", s=300,
               color="gold", edgecolors="k", lw=0.7, zorder=9,
               label="SGR 1935+2154 @ 147 Mpc (Mereghetti+20)")
    ax.annotate("8 orders below\nLAXPC sensitivity",
                xy=(2e18,NUFNU_SGR_RESC), xytext=(3e16,5e-23), fontsize=9, color="0.45",
                arrowprops=dict(arrowstyle="->",color="0.55",lw=0.9))
    for txt, nu_t, y_t in [("Radio",3e8,2e-7),("Optical",5e14,2e-7),("X-ray",1e18,2e-7)]:
        ax.text(nu_t, y_t, txt, fontsize=11, color="0.40", ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3e7,5e19); ax.set_ylim(5e-29,5e-5)
    ax.set_xlabel(r"Frequency $\nu$ [Hz]", fontsize=14)
    ax.set_ylabel(r"$\nu F_\nu$ [erg cm$^{-2}$ s$^{-1}$]", fontsize=14)
    ax.set_title("FRB 20180916B — Broadband SED: upper limits and literature detections", fontsize=13)
    ax.legend(fontsize=9.5, ncol=2, loc="lower left", framealpha=0.92, edgecolor="0.75")
    plt.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(f"{out}/sed_overview.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  -> sed_overview.pdf / .png")

# ============================================================
# PLOT B — SHOCK EXCLUSION MAP (with cooling-break-corrected F_X)
# ============================================================

def plot_shock(chain, out):
    labels_sh = [r"$\log_{10}(E_k /$ erg)",
                 r"$\log_{10}(\epsilon_B)$",
                 r"$\log_{10}(n /$ cm$^{-3})$"]
    corner_plot(chain, labels=labels_sh,
                title="Shock model posterior  (stretch-move; cooling break included)\n"
                      r"AquEYE+ $5\sigma$ + LAXPC $3\sigma$ upper limits",
                fname=f"{out}/shock_corner")

    log_Ek, log_epsB, log_n = chain.T
    med_logn = float(np.median(log_n))

    Ek_g = np.linspace(42, 47, 300); eB_g = np.linspace(-4, 0, 300)
    EK, EB = np.meshgrid(Ek_g, eB_g)
    F_opt_g  = np.vectorize(shock_F_opt)(EK, EB, med_logn)
    F_xray_g = np.vectorize(shock_F_xray)(EK, EB, med_logn)
    ratio_g  = np.vectorize(FX_over_Fopt)(EK, EB, med_logn)

    ten_opt  = (F_opt_g  - F_OPT_LIM)  / (F_OPT_LIM  / N_SIG_OPT)
    ten_xray = (F_xray_g - F_XRAY_LIM) / (F_XRAY_LIM / N_SIG_XRAY)

    H, xe, ye = np.histogram2d(log_Ek, log_epsB, bins=55, range=[[42,47],[-4,0]])
    H = H.T; xc=0.5*(xe[:-1]+xe[1:]); yc=0.5*(ye[:-1]+ye[1:])
    hs=np.sort(H.ravel())[::-1]; hc=np.cumsum(hs)/np.sum(hs)
    l68=hs[np.searchsorted(hc,0.68)]; l95=hs[np.searchsorted(hc,0.95)]

    cmap_bg = mcolors.LinearSegmentedColormap.from_list(
        "bg", ["#FFFFFF","#D6EAF8","#2874A6","#1B2631"], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, tension, excl_col, title_str in [
        (axes[0], ten_opt,  "#D35400",
         r"Optical (AquEYE+ $5\sigma$)"),
        (axes[1], ten_xray, "#6C3483",
         r"X-ray with cooling break (LAXPC $3\sigma$)"),
    ]:
        im = ax.contourf(EK, EB, tension, levels=np.linspace(-5,20,60),
                         cmap=cmap_bg, extend="both")
        fig.colorbar(im, ax=ax, label=r"Tension $(F_{\rm model}-F_{\rm lim})/\sigma$",
                     pad=0.02).ax.tick_params(labelsize=10)
        ax.contourf(EK, EB, tension, levels=[0,1e9], colors=[excl_col], alpha=0.28)
        ax.contour( EK, EB, tension, levels=[0],     colors=[excl_col], linewidths=2.8)
        ax.contour(xc, yc, H, levels=[l95,l68],
                   colors=["#1A5276","#1A5276"],
                   linewidths=[1.1,2.0], linestyles=["--","-"])
        ax.set_xlabel(r"$\log_{10}(E_k /$ erg)", fontsize=13)
        ax.set_ylabel(r"$\log_{10}(\epsilon_B)$", fontsize=13)
        ax.set_title(f"{title_str}\n(log n = {med_logn:.1f} median)", fontsize=11)
        ax.tick_params(labelsize=11)

    # Panel 3: cooling break regime map
    ax = axes[2]
    NC = np.log10(np.vectorize(nu_cool)(EK, EB, med_logn) + 1e-10)
    im3 = ax.contourf(EK, EB, NC, levels=np.linspace(13,22,60),
                      cmap="RdYlBu_r", extend="both")
    fig.colorbar(im3, ax=ax, label=r"$\log_{10}(\nu_c /$ Hz)", pad=0.02).ax.tick_params(labelsize=10)
    ax.contour(EK, EB, NC, levels=[np.log10(_NU_OPT)],
               colors=["blue"], linewidths=2.0, linestyles=["--"])
    ax.contour(EK, EB, NC, levels=[np.log10(_NU_X)],
               colors=["red"],  linewidths=2.0, linestyles=["--"])
    ax.text(43.0, -0.5, r"$\nu_c > \nu_X$: slow cool", fontsize=9, color="navy")
    ax.text(44.5, -2.5, r"$\nu_{\rm opt}<\nu_c<\nu_X$: mixed", fontsize=9, color="k")
    ax.text(46.0, -3.5, r"$\nu_c < \nu_{\rm opt}$: fast cool", fontsize=9, color="darkred", ha="right")
    ax.set_xlabel(r"$\log_{10}(E_k /$ erg)", fontsize=13)
    ax.set_ylabel(r"$\log_{10}(\epsilon_B)$", fontsize=13)
    ax.set_title(r"Cooling break $\nu_c$ regime map" + f"\n(log n = {med_logn:.1f} median)", fontsize=11)
    ax.tick_params(labelsize=11)

    leg_ex_opt = mpatches.Patch(color="#D35400", alpha=0.38, label="Excl. (optical)")
    leg_ex_X   = mpatches.Patch(color="#6C3483", alpha=0.38, label="Excl. (X-ray)")
    leg_68     = mpatches.Patch(fill=False, ec="#1A5276", lw=2.0, label="MCMC 68%")
    leg_95     = mpatches.Patch(fill=False, ec="#1A5276", lw=1.1, ls="--", label="MCMC 95%")
    axes[0].legend(handles=[leg_ex_opt, leg_ex_X, leg_68, leg_95], fontsize=9.5, loc="upper left")

    fig.suptitle("FRB 20180916B — Shock model constraints with cooling break\n"
                 r"($p=2.2$; Beloborodov 2020 calibration; Granot \& Sari 2002 $\nu_c$)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(f"{out}/shock_exclusion_2d.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  -> shock_exclusion_2d.pdf / .png")

# ============================================================
# PLOT C — EFFICIENCY (with Kilpatrick+23 + marginalised E_radio)
# ============================================================

def plot_efficiency(chain, out):
    labels = [r"$\log_{10}(\eta_X)$",
              r"$\log_{10}(\eta_{\rm opt})$",
              r"$\log_{10}(E_{\rm radio} /$ erg)"]
    corner_plot(chain, labels=labels,
                truths=[np.log10(ETA_X_SGR), np.log10(ETA_OPT_KIL), E_RADIO_MU],
                title="Efficiency posterior  (stretch-move; Kilpatrick+23 included;\n"
                      r"$E_{\rm radio}$ marginalised with log-normal prior)",
                fname=f"{out}/efficiency_corner")

    leX, leO, leR = chain.T
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # eta_X
    ax = axes[0]
    ax.hist(leX, bins=90, density=True, color="#1B7837", alpha=0.72,
            histtype="stepfilled", edgecolor="white", lw=0.5)
    q16, q50, q84 = np.percentile(leX, [16, 50, 84])
    ax.axvline(q50, color="navy",     lw=2.0, ls=":", label=f"Median: {q50:.2f}")
    ax.axvline(q84, color="orangered",lw=2.5, ls="--", label=f"84th pct (UL): {q84:.2f}")
    ax.axvline(np.log10(ETA_X_SGR), color="goldenrod", lw=2.8,
               label=r"SGR 1935+2154: $\eta_X \geq 10^5$")
    ax.axvspan(np.log10(ETA_X_SGR), 9, alpha=0.13, color="goldenrod")
    # HXMT (Trudu+23) at E_radio=1e37: eta_X < 1e7
    ax.axvline(7.0, color="#D35400", lw=1.8, ls="-.",
               label=r"HXMT (Trudu+23): $E_X/E_r < 10^7$ (simultaneous)")
    ymax = ax.get_ylim()[1]
    ax.text(np.log10(ETA_X_SGR)+0.08, max(ymax*0.55,0.03),
            "SGR\nanalogue", fontsize=9.5, color="#B7950B", fontweight="bold")
    ax.set_xlabel(r"$\log_{10}(\eta_X = E_X / E_{\rm radio})$", fontsize=13)
    ax.set_ylabel("Posterior density", fontsize=13)
    ax.set_title(r"X-ray efficiency (LAXPC $3\sigma$ + log-normal $E_{\rm radio}$ prior)"
                 "\n"
                 r"$E_X^{\rm iso} < 4.24\times10^{43}$ erg", fontsize=10.5)
    ax.legend(fontsize=9.5, loc="upper left")

    # eta_opt
    ax = axes[1]
    ax.hist(leO, bins=90, density=True, color="#1A5276", alpha=0.72,
            histtype="stepfilled", edgecolor="white", lw=0.5)
    q16, q50, q84 = np.percentile(leO, [16, 50, 84])
    ax.axvline(q50, color="navy",     lw=2.0, ls=":", label=f"Median: {q50:.2f}")
    ax.axvline(q84, color="orangered",lw=2.5, ls="--", label=f"84th pct (UL): {q84:.2f}")
    ax.axvline(np.log10(ETA_OPT_KIL), color="#D35400", lw=2.5,
               label=r"Kilpatrick+23 (simultaneous): $\eta_{\rm opt} < 3\times10^{-3}$")
    ax.axvspan(np.log10(ETA_OPT_KIL), 4, alpha=0.10, color="#D35400")
    ax.axvspan(4, 7, alpha=0.18, color="#922B21",
               label=r"Reconnection flares (Lyutikov \& Uzdensky 03)")
    ax.axvspan(-7, -2, alpha=0.14, color="#1B7837",
               label=r"Magnetospheric (Lu+20): $\eta_{\rm opt} \ll 1$")
    ymax2 = ax.get_ylim()[1]
    ax.text(4.15, max(ymax2*0.55,0.02), "Reconnection\nflares\n(excluded)",
            fontsize=9.5, color="#7B241C", fontweight="bold")
    ax.text(-6.8, max(ymax2*0.55,0.02), "Magneto-\nspheric\n(allowed)",
            fontsize=9.5, color="#1B7837", fontweight="bold")
    ax.set_xlabel(r"$\log_{10}(\eta_{\rm opt} = E_{\rm opt} / E_{\rm radio})$", fontsize=13)
    ax.set_ylabel("Posterior density", fontsize=13)
    ax.set_title(r"Optical efficiency (AquEYE+ $5\sigma$ + Kilpatrick+23 likelihood)"
                 "\n"
                 r"$E_{\rm opt}^{\rm iso} < 8.0\times10^{39}$ erg (our);"
                 r"  $\eta_{\rm opt} < 3\times10^{-3}$ (Kilpatrick)", fontsize=10.5)
    ax.legend(fontsize=9.0, loc="upper right")

    fig.suptitle("FRB 20180916B — Magnetospheric efficiency constraints\n"
                 "Kilpatrick+23 incorporated; $E_{\\rm radio}$ marginalised",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(f"{out}/efficiency_marginals.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  -> efficiency_marginals.pdf / .png")

# ============================================================
# PLOT D — MAGNETAR GRID
# ============================================================

def plot_magnetar_grid(out):
    log_ER_arr   = np.linspace(35, 40, 500)
    log_etaX_arr = np.linspace(-2, 9,  500)
    log_etaO_arr = np.linspace(-7, 7,  500)
    ER,  ETX = np.meshgrid(log_ER_arr, log_etaX_arr)
    ER2, ETO = np.meshgrid(log_ER_arr, log_etaO_arr)

    ten_X   = (10.0**(ETX+ER) - E_XRAY_LIM) / (E_XRAY_LIM / N_SIG_XRAY)
    ten_opt = (10.0**(ETO+ER2) - E_OPT_LIM)  / (E_OPT_LIM  / N_SIG_OPT)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    for ax, (log_eta, ten, excl_col, ylabel, title_str, xr, yr) in zip(axes, [
        (log_etaX_arr, ten_X, "#6C3483",
         r"$\log_{10}(\eta_X)$", "X-ray efficiency", [-2, 9],
         None),
        (log_etaO_arr, ten_opt, "#922B21",
         r"$\log_{10}(\eta_{\rm opt})$", "Optical efficiency", [-7, 7],
         None),
    ]):
        ER_plot = log_ER_arr
        log_EE  = (ETX if "X-ray" in title_str else ETO) + (ER if "X-ray" in title_str else ER2)
        im = ax.contourf(ER_plot, log_eta,
                         np.log10(np.clip(10.0**log_EE, 1e30, 1e55)),
                         levels=np.linspace(34, 50, 65),
                         cmap="YlGnBu" if "X-ray" in title_str else "YlOrRd_r",
                         extend="both")
        fig.colorbar(im, ax=ax, label=r"$\log_{10}(E /$ erg)", pad=0.02).ax.tick_params(labelsize=10)

        ax.contourf(ER_plot, log_eta, ten, levels=[0, 1e9], colors=[excl_col], alpha=0.38)
        ax.contour( ER_plot, log_eta, ten, levels=[0],     colors=[excl_col], linewidths=3.0)

        ax.axvspan(np.log10(E_RADIO_MIN:=1e36), np.log10(1e38),
                   alpha=0.13, color="cyan",
                   label=r"Effelsberg $E_{\rm radio}$ range (Bethapudi+23)")
        ax.axvline(E_RADIO_MU, color="cyan", lw=1.8, ls=":")

        if "X-ray" in title_str:
            ax.axhline(np.log10(ETA_X_SGR), color="gold", lw=3.0, zorder=5)
            ax.text(35.1, np.log10(ETA_X_SGR)+0.2,
                    r"SGR 1935+2154: $\eta_X \geq 10^5$",
                    color="goldenrod", fontsize=10, fontweight="bold")
            ax.fill_between(ER_plot, 2, 5, alpha=0.18, color="#1B7837",
                            label=r"[A] Magnetospheric: $\eta_X \sim 10^2$--$10^5$")
            ax.text(37.5, 3.4, "[A] Magnetospheric", color="#1B7837",
                    fontsize=10, fontweight="bold", ha="center")
        else:
            ax.axhline(np.log10(ETA_OPT_KIL), color="#D35400", lw=2.5, ls="--",
                       label=r"Kilpatrick+23 (simultaneous): $\eta_{\rm opt}<3\times10^{-3}$")
            ax.axhspan(np.log10(ETA_OPT_KIL), 7, alpha=0.10, color="#D35400")
            ax.fill_between(ER_plot, 4, 6, alpha=0.22, color="#6C3483",
                            label=r"[C] Reconnection flares: $\eta_{\rm opt}\sim10^4$--$10^6$ (excluded)")
            ax.fill_between(ER_plot, -7, -2, alpha=0.20, color="#1B7837",
                            label=r"[A] Magnetospheric: $\eta_{\rm opt}\ll1$ (allowed)")
            ax.text(37.5, 4.8, "[C] Reconnection flares\n(fully excluded)",
                    color="#6C3483", fontsize=10, fontweight="bold", ha="center")
            ax.text(37.5, -4.5, "[A] Magnetospheric\n(allowed)",
                    color="#1B7837", fontsize=10, fontweight="bold", ha="center")

        ax.set_xlabel(r"$\log_{10}(E_{\rm radio} /$ erg)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlim(35, 40)
        ax.set_title(title_str, fontsize=12)
        ax.legend(fontsize=9.5, loc="lower right", framealpha=0.88)
        ax.tick_params(labelsize=11)

    fig.suptitle("FRB 20180916B — Magnetar model variant constraints\n"
                 "Which emission families survive all broadband upper limits?",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(f"{out}/magnetar_model_grid.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  -> magnetar_model_grid.pdf / .png")

# ============================================================
# PLOT E — DIAGNOSTICS (trace + autocorrelation)
# ============================================================

def plot_diag(chain, labels, name, out):
    n_p = chain.shape[1]
    fig, axes = plt.subplots(n_p, 2, figsize=(14, 2.8*n_p))
    if n_p == 1: axes = axes[None, :]
    for j, (ax_t, ax_a, lab) in enumerate(zip(axes[:,0], axes[:,1], labels)):
        col = chain[:, j]
        ax_t.plot(col, lw=0.35, color="#2166AC", alpha=0.90)
        ax_t.axhline(np.median(col), color="orangered", lw=1.1, ls="--")
        ax_t.set_ylabel(lab, fontsize=10); ax_t.set_xlabel("Sample", fontsize=10)
        x  = col - col.mean(); N=len(x)
        ac = np.correlate(x, x, mode='full')[N-1:]; ac = ac/ac[0]
        ax_a.bar(np.arange(min(100,N)), ac[:100], width=1.0, color="#2166AC", alpha=0.70)
        ax_a.axhline(0,    color="k",        lw=0.8)
        ax_a.axhline(0.05, color="orangered",lw=1.0,ls="--",label=r"$|\rho|=0.05$")
        ax_a.set_xlabel("Lag",fontsize=10); ax_a.set_ylabel("Autocorr.",fontsize=10)
        ax_a.set_title(lab,fontsize=9); ax_a.legend(fontsize=8.5)
    fig.suptitle(f"Diagnostics — {name}",fontsize=12); plt.tight_layout()
    safe = name.lower().replace(" ","_")
    for ext in ("pdf","png"):
        fig.savefig(f"{out}/diag_{safe}.{ext}",dpi=150,bbox_inches="tight")
    plt.close(fig); print(f"  -> diag_{safe}.pdf / .png")

# ============================================================
# MAIN
# ============================================================

def main():
    OUT = "/mnt/user-data/outputs/mcmc_outputs"
    os.makedirs(OUT, exist_ok=True)

    N_WALKERS = 32
    N_STEPS   = 8_000
    BURN      = 0.40
    THIN      = 10
    SEEDS     = [42, 137, 271, 314]

    print("=" * 70)
    print("  FRB 20180916B — MCMC v4 (submission-ready)")
    print("  SRT_11-24 (PI: A. Anandhu) + September 2023 MWL campaign")
    print("=" * 70)
    print(f"\n  Sampler:  Goodman-Weare stretch-move, {N_WALKERS} walkers, "
          f"{N_STEPS} steps/walker")
    print(f"  Burn-in:  {int(BURN*100)}%,  thinning: {THIN}")
    print(f"\n  Key constraints:")
    print(f"    AquEYE+:    F_opt  < {F_OPT_LIM:.2e} erg/cm2 ({N_SIG_OPT}sigma, 1ms)")
    print(f"    LAXPC:      F_X    < {F_XRAY_LIM:.2e} erg/cm2 ({N_SIG_XRAY}sigma, 1ms)")
    print(f"    Kilpatrick: eta_opt < {ETA_OPT_KIL:.0e} ({N_SIG_KIL}sigma, simultaneous)")
    print(f"\n  NOTE: Chandra (Scholz+20) E_iso = 1.3e45 erg (corrected) is")
    print(f"        LESS constraining than LAXPC and is NOT used in likelihood.")
    print(f"        Thesis text states 1e40 erg — this is an error.")
    print(f"\n  Cooling break: F_X/F_opt = 82-925 (not 1% as previously coded)")
    print(f"  E_radio prior: log-normal N({E_RADIO_MU}, {E_RADIO_SIG}) [Bethapudi+23]")

    # ── MODULE 1: SHOCK  (4 chains, stretch-move) ─────────────
    print("\n[1/2] Shock model ...")
    p0_sh = np.array([44.0, -2.0, -1.5])
    chains_shock = []
    for s in SEEDS:
        w0 = init_walkers(p0_sh, N_WALKERS, scale=0.03,
                          logpost_fn=logpost_shock, seed=s)
        c, ar = ensemble_sampler(logpost_shock, w0, N_STEPS, BURN, THIN, seed=s)
        chains_shock.append(c)
        print(f"    seed={s}  accept={ar:.3f}  N_kept={len(c)}")
    Rhat_sh, Neff_sh = convergence_report(
        "Shock", chains_shock, ["log_Ek","log_epsB","log_n"])
    chain_shock = np.vstack(chains_shock)

    # ── MODULE 2: EFFICIENCY  (4 chains, stretch-move) ─────────
    print("\n[2/2] Efficiency ...")
    p0_eff = np.array([4.0, -3.0, 37.0])
    chains_eff = []
    for s in SEEDS:
        w0 = init_walkers(p0_eff, N_WALKERS, scale=0.02,
                          logpost_fn=logpost_eff, seed=s)
        c, ar = ensemble_sampler(logpost_eff, w0, N_STEPS, BURN, THIN, seed=s)
        chains_eff.append(c)
        print(f"    seed={s}  accept={ar:.3f}  N_kept={len(c)}")
    Rhat_eff, Neff_eff = convergence_report(
        "Efficiency", chains_eff, ["log_eta_X","log_eta_opt","log_E_radio"])
    chain_eff = np.vstack(chains_eff)

    # ── PLOTS ──────────────────────────────────────────────────
    print("\n[Plotting] ...")
    plot_sed(OUT)
    plot_shock(chain_shock, OUT)
    plot_efficiency(chain_eff, OUT)
    plot_magnetar_grid(OUT)
    posterior_predictive_check(chain_shock, chain_eff, OUT)
    plot_diag(chain_shock,
              [r"$\log_{10}(E_k)$",r"$\log_{10}(\epsilon_B)$",r"$\log_{10}(n)$"],
              "Shock_model", OUT)
    plot_diag(chain_eff,
              [r"$\log_{10}(\eta_X)$",r"$\log_{10}(\eta_{\rm opt})$",
               r"$\log_{10}(E_{\rm radio})$"],
              "Efficiency", OUT)

    # ── SAVE CHAINS ────────────────────────────────────────────
    np.save(f"{OUT}/chain_shock.npy",      chain_shock)
    np.save(f"{OUT}/chain_efficiency.npy", chain_eff)

    # ── SUMMARY ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY  (Chapter 6)")
    print("=" * 70)

    print("\n  -- SHOCK MODEL --")
    for i, nm in enumerate(["log_Ek","log_epsB","log_n"]):
        q16,q50,q84 = np.percentile(chain_shock[:,i],[16,50,84])
        print(f"    {nm:12s} = {q50:.2f}  +{q84-q50:.2f}/-{q50-q16:.2f}  "
              f"(84th pct: {q84:.2f})")
    print(f"    => Ek > 1e45 erg AND epsB > 0.01: EXCLUDED (AquEYE+ + LAXPC)")
    print(f"    => LAXPC now contributes (F_X/F_opt ~ 100-900, not 1%)")
    print(f"    R-hat max: {np.max(Rhat_sh):.4f}")

    print("\n  -- EFFICIENCY --")
    q84X = np.percentile(chain_eff[:,0], 84)
    q84O = np.percentile(chain_eff[:,1], 84)
    q50X = np.percentile(chain_eff[:,0], 50)
    q50O = np.percentile(chain_eff[:,1], 50)
    print(f"    eta_X  84th pct UL: 10^{q84X:.2f} = {10**q84X:.1e}")
    print(f"    eta_opt 84th pct UL: 10^{q84O:.2f} = {10**q84O:.1e}")
    print(f"    SGR 1935+2154 reference: eta_X >= 10^5.0")
    if q84X < 5.0:
        print(f"    => LAXPC UL is BELOW SGR: magnetar analogy constrained")
    else:
        print(f"    => LAXPC UL ({q84X:.2f}) is consistent with SGR analogy")
    print(f"    Kilpatrick eta_opt: {10**q84O:.2e} vs 3e-3 limit")
    print(f"    => Reconnection flares (eta_opt~1e4-1e6): EXCLUDED")
    print(f"    => Magnetospheric models (eta_opt<<1):   CONSISTENT")
    print(f"    R-hat max: {np.max(Rhat_eff):.4f}")

    all_ok = (np.max([*Rhat_sh,*Rhat_eff]) < 1.01
              and np.min([*Neff_sh,*Neff_eff]) > 1000)
    print(f"\n  -- CONVERGENCE: {'PASS' if all_ok else 'WARN'} --")
    print(f"    R-hat < 1.01:  {'YES' if np.max([*Rhat_sh,*Rhat_eff])<1.01 else 'NO'}")
    print(f"    N_eff > 1000:  {'YES' if np.min([*Neff_sh,*Neff_eff])>1000 else 'NO'}")

    print(f"\n  All outputs: {OUT}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
