"""
MCMC Analysis for FRB 20180916B — Chapter 6  (v4)
===================================================
Three datasets compared side-by-side:
  A) THIS WORK    — SRT/uGMRT, AquEYE+, LAXPC
  B) LITERATURE   — LOFAR, CHIME, Effelsberg, Chandra, Gemini/'Alopeke
  C) COMBINED     — A + B together

Three analyses per dataset:
  1. Radio spectral index  α  (F_ν ∝ ν^{-α})
  2. Shock model  (log E_k, log ε_B)
  3. Magnetospheric efficiencies  (log η_opt, log η_X)

Step sizes tuned for 20–40% acceptance rate.
All values cited with table/chapter/paper references.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.special import erfc

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# METROPOLIS-HASTINGS  (tuned step sizes)
# ─────────────────────────────────────────────────────────────
def mh_sampler(log_prob_fn, initial, n_steps=300000, step_size=None):
    ndim    = len(initial)
    if step_size is None:
        step_size = np.ones(ndim) * 0.3   # larger default → lower accept rate
    chain      = np.zeros((n_steps, ndim))
    current    = np.array(initial, dtype=float)
    lp_current = log_prob_fn(current)
    n_acc      = 0
    for i in range(n_steps):
        prop    = current + np.random.randn(ndim) * step_size
        lp_prop = log_prob_fn(prop)
        if np.log(np.random.uniform()) < lp_prop - lp_current:
            current    = prop
            lp_current = lp_prop
            n_acc     += 1
        chain[i] = current
    rate = n_acc / n_steps
    flag = "✓" if 0.15 < rate < 0.55 else "⚠ CHECK"
    print(f"    Acceptance: {rate:.1%}  {flag}")
    return chain

def flat_samples(chain, burnin=0.40, thin=25):
    b = int(len(chain) * burnin)
    return chain[b::thin]

def percentiles(s, p=[16, 50, 84]):
    return np.percentile(s, p)


# ─────────────────────────────────────────────────────────────
# CORNER PLOT  (single dataset)
# ─────────────────────────────────────────────────────────────
def corner_single(samples, labels, title, outfile,
                  color='steelblue', xlims=None,
                  ref_lines=None):
    """ref_lines: list of (axis, idx, value, color, ls, label)
       axis='v' or 'h', idx = col or row index"""
    nd  = samples.shape[1]
    fig, axes = plt.subplots(nd, nd, figsize=(4.2*nd, 4.2*nd))
    fig.suptitle(title, fontsize=11, y=0.99)

    for row in range(nd):
        for col in range(nd):
            ax = axes[row, col] if nd > 1 else axes

            if col > row:
                ax.set_visible(False)
                continue

            if col == row:
                s = samples[:, col]
                ax.hist(s, bins=60, color=color, alpha=0.50,
                        density=True, histtype='stepfilled')
                ax.hist(s, bins=60, color=color, alpha=1.0,
                        density=True, histtype='step', lw=1.2)
                q16, q50, q84 = percentiles(s)
                lo, hi = q50-q16, q84-q50
                for q, lw in zip([q16,q50,q84],[1.0,1.8,1.0]):
                    ax.axvline(q, color=color, lw=lw,
                               ls=':' if lw==1.0 else '-')
                ts = (f"{labels[col]}\n"
                      f"${q50:.2f}"
                      r"^{+" + f"{hi:.2f}" + r"}_{-"
                      + f"{lo:.2f}" + r"}$")
                ax.set_title(ts, fontsize=9)
                ax.set_yticks([])
                ax.set_xlabel(labels[col], fontsize=10)
                if xlims and col < len(xlims) and xlims[col]:
                    ax.set_xlim(xlims[col])
                if ref_lines:
                    for ax2, idx, val, rc, rls, rlbl in ref_lines:
                        if ax2 == 'v' and idx == col:
                            ax.axvline(val, color=rc, lw=1.8,
                                       ls=rls, label=rlbl)
                            ax.legend(fontsize=7)
                continue

            x, y = samples[:,col], samples[:,row]
            h, xe, ye = np.histogram2d(x, y, bins=45)
            h  = gaussian_filter(h.T, sigma=1.2)
            hf = np.sort(h.ravel())[::-1]
            cm = np.cumsum(hf)/np.sum(hf)
            l68 = hf[np.searchsorted(cm, 0.68)]
            l95 = hf[np.searchsorted(cm, 0.95)]
            X   = 0.5*(xe[:-1]+xe[1:])
            Y   = 0.5*(ye[:-1]+ye[1:])
            ax.contourf(X,Y,h, levels=[l95,l68,h.max()+1],
                        colors=[color,color], alpha=[0.18,0.42])
            ax.contour(X,Y,h, levels=[l95,l68],
                       colors=[color], linewidths=1.0)
            ax.set_xlabel(labels[col], fontsize=10)
            ax.set_ylabel(labels[row], fontsize=10)
            if xlims:
                if col < len(xlims) and xlims[col]:
                    ax.set_xlim(xlims[col])
                if row < len(xlims) and xlims[row]:
                    ax.set_ylim(xlims[row])
            if ref_lines:
                for ax2, idx, val, rc, rls, rlbl in ref_lines:
                    if ax2 == 'v' and idx == col:
                        ax.axvline(val, color=rc, lw=1.8, ls=rls,
                                   label=rlbl)
                    if ax2 == 'h' and idx == row:
                        ax.axhline(val, color=rc, lw=1.8, ls=rls,
                                   label=rlbl)
                ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {outfile}")


# ═══════════════════════════════════════════════════════════════
#  DATA DEFINITIONS
# ═══════════════════════════════════════════════════════════════

nu_ref = 1e9   # 1 GHz pivot
D_L    = 147e6 * 3.086e18  # cm  (147 Mpc)

# ── THIS WORK ───────────────────────────────────────────────
# Detections: SRT 328 MHz, Table 3.19
tw_det_nu  = np.array([328e6, 328e6, 328e6, 328e6])
tw_det_F   = np.array([1.9, 6.2, 1.9, 2.9])     # Jy ms
tw_det_sig = np.array([0.20, 0.20, 0.20, 0.20])  # dex

# Upper limits: SRT Table 3.14
tw_ul_nu   = np.array([4.6e9,  6.5e9, 19.0e9])
tw_ul_F    = np.array([0.30,   0.30,   4.00])    # Jy ms

# Multiwavelength constraints (Chapters 4 & 5)
tw_Fopt_lim = 3.1e-15    # erg/cm2, AquEYE+ 5σ 1ms, Ch.4
tw_Eopt_lim = 8.1e39     # erg,     AquEYE+ iso, Ch.4
tw_FX_lim   = 3.76e-11   # erg/cm2, LAXPC 3σ 1ms, Ch.5
tw_EX_lim   = 9.7e43     # erg,     LAXPC iso, Ch.5
tw_E_radio  = np.median([3.0e36,1.0e37,3.0e36,4.6e36])  # Table 3.19

# ── LITERATURE ───────────────────────────────────────────────
# Detections with known ranges
lit_det_nu  = np.array([150e6,  600e6,  6000e6])
lit_det_Flo = np.array([5.0,    1.0,    0.08  ])
lit_det_Fhi = np.array([50.0,   10.0,   1.49  ])
lit_det_F   = np.sqrt(lit_det_Flo * lit_det_Fhi)   # geom. mean
lit_det_sig = 0.5*(np.log10(lit_det_Fhi) -
                   np.log10(lit_det_Flo))            # dex half-range
# Sources: LOFAR (Pleunis2020,Gopinath2024),
#          CHIME (CHIME2019,2021), Effelsberg (Bethapudi2022,2023)

# Optical: Gemini/'Alopeke (Kilpatrick2023)
lit_Fopt_lim = 8.3e-3 * 1e-26 * 8.73e13 * 1e-2  # ~8.3e-3 Jy ms → erg/cm2
# Convert: 8.3e-3 Jy ms = 8.3e-3 * 1e-26 J/(m2 Hz) * BW * dt
# More precisely from thesis: ~ 1e-12 erg/cm2
lit_Fopt_lim = 1.0e-12   # erg/cm2, Kilpatrick2023
lit_Eopt_lim = 4*np.pi*D_L**2 * lit_Fopt_lim  # erg

# X-ray: Chandra (Scholz2020) — most constraining literature X-ray
lit_FX_lim   = 5.0e-10   # erg/cm2, Chandra 0.5-10 keV
lit_EX_lim   = 1.0e40    # erg (from thesis §6.4.3)

# Literature E_radio reference: typical Effelsberg burst
lit_E_radio  = 1.3e37    # erg, thesis §6.1.5

# ── COMBINED ────────────────────────────────────────────────
# Detections: SRT (this work) + LOFAR,CHIME,Effelsberg (lit)
comb_det_nu  = np.concatenate([tw_det_nu,  lit_det_nu ])
comb_det_F   = np.concatenate([tw_det_F,   lit_det_F  ])
comb_det_sig = np.concatenate([tw_det_sig, lit_det_sig])

# Upper limits: SRT (this work) — same bands
comb_ul_nu   = tw_ul_nu
comb_ul_F    = tw_ul_F

# Best constraint per band (most constraining from either source):
comb_Fopt_lim = tw_Fopt_lim  # AquEYE+ beats Gemini by ~3 orders of magnitude
comb_Eopt_lim = tw_Eopt_lim
comb_FX_lim   = lit_FX_lim   # Chandra beats LAXPC for soft X-ray
comb_EX_lim   = lit_EX_lim
comb_E_radio  = tw_E_radio   # use our measured value


# ═══════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

sigma_ul = 0.15   # dex softness for upper limits

def make_log_prob_spectral(det_nu, det_F, det_sig, ul_nu, ul_F):
    def log_prob(theta):
        log_A, alpha = theta
        if not (-0.5 < log_A < 3.5 and 0.3 < alpha < 6.0):
            return -np.inf
        # Detections
        logFm = log_A - alpha * np.log10(det_nu / nu_ref)
        chi2  = np.sum(((np.log10(det_F) - logFm) / det_sig)**2)
        ll    = -0.5 * chi2
        # Upper limits
        for nu_i, F_lim in zip(ul_nu, ul_F):
            logFm_ul = log_A - alpha * np.log10(nu_i / nu_ref)
            log_Fl   = np.log10(F_lim)
            ll += np.log(0.5 * erfc((logFm_ul - log_Fl)
                                     / (np.sqrt(2)*sigma_ul)) + 1e-300)
        return ll
    return log_prob

# Shock model calibration (thesis §6.4.4)
F_opt_calib = 1e-13; E_k_ref = 1e45; epsB_ref = 0.01
a_Ek = 1.05; a_epsB = 0.10

def F_shock_pred(log_Ek, log_epsB):
    return (F_opt_calib
            * (10**log_Ek  / E_k_ref )**a_Ek
            * (10**log_epsB / epsB_ref)**a_epsB)

def make_log_prob_shock(Fopt_lim):
    sig = 0.20
    def log_prob(theta):
        log_Ek, log_epsB = theta
        if not (42 < log_Ek < 48 and -5 < log_epsB < 0):
            return -np.inf
        lFp = np.log10(F_shock_pred(log_Ek, log_epsB))
        lFl = np.log10(Fopt_lim)
        if lFp <= lFl:
            return 0.0
        return -0.5 * ((lFp - lFl) / sig)**2
    return log_prob

def make_log_prob_mag(Eopt_lim, EX_lim, E_radio):
    eta_o_lim = Eopt_lim / E_radio
    eta_x_lim = EX_lim  / E_radio
    sig = 0.15
    def log_prob(theta):
        lo, lx = theta
        if not (-15 < lo < 12 and -15 < lx < 14):
            return -np.inf
        ll = 0.0
        if lo > np.log10(eta_o_lim):
            ll -= 0.5*((lo - np.log10(eta_o_lim))/sig)**2
        if lx > np.log10(eta_x_lim):
            ll -= 0.5*((lx - np.log10(eta_x_lim))/sig)**2
        return ll
    return log_prob


# ═══════════════════════════════════════════════════════════════
# RUN ALL THREE DATASETS × THREE ANALYSES
# ═══════════════════════════════════════════════════════════════

datasets = {
    'This Work':  {
        'color': '#2166ac',
        'det_nu': tw_det_nu,  'det_F': tw_det_F,  'det_sig': tw_det_sig,
        'ul_nu':  tw_ul_nu,   'ul_F':  tw_ul_F,
        'Fopt': tw_Fopt_lim, 'Eopt': tw_Eopt_lim,
        'FX':   tw_FX_lim,   'EX':   tw_EX_lim,
        'Erad': tw_E_radio,
        'spec_label': 'SRT detections + SRT limits\n(Tables 3.14, 3.19)',
        'opt_label':  r'AquEYE+ $5\sigma$ 1ms (Ch.4)',
        'X_label':    r'LAXPC $3\sigma$ 1ms (Ch.5)',
    },
    'Literature': {
        'color': '#d73027',
        'det_nu': lit_det_nu, 'det_F': lit_det_F, 'det_sig': lit_det_sig,
        'ul_nu':  tw_ul_nu,   'ul_F':  tw_ul_F,   # same upper limits available
        'Fopt': lit_Fopt_lim, 'Eopt': lit_Eopt_lim,
        'FX':   lit_FX_lim,   'EX':   lit_EX_lim,
        'Erad': lit_E_radio,
        'spec_label': 'LOFAR/CHIME/Effelsberg detections\n(Pleunis+20, CHIME+21, Bethapudi+23)',
        'opt_label':  r"Gemini/'Alopeke $3\sigma$ (Kilpatrick+23)",
        'X_label':    r'Chandra $3\sigma$ (Scholz+20)',
    },
    'Combined':   {
        'color': '#1a9641',
        'det_nu': comb_det_nu, 'det_F': comb_det_F, 'det_sig': comb_det_sig,
        'ul_nu':  comb_ul_nu,  'ul_F':  comb_ul_F,
        'Fopt': comb_Fopt_lim, 'Eopt': comb_Eopt_lim,
        'FX':   comb_FX_lim,   'EX':   comb_EX_lim,
        'Erad': comb_E_radio,
        'spec_label': 'All detections + SRT limits\n(this work + literature)',
        'opt_label':  r'AquEYE+ $5\sigma$ (deepest, this work)',
        'X_label':    r'Chandra (most constraining, Scholz+20)',
    },
}

results = {}

for dname, d in datasets.items():
    print(f"\n{'='*55}")
    print(f"  Dataset: {dname}")
    print(f"{'='*55}")

    # ── Spectral index ──
    print("  [1] Spectral index")
    lp_sp = make_log_prob_spectral(
        d['det_nu'], d['det_F'], d['det_sig'],
        d['ul_nu'],  d['ul_F'])
    # Tune step size based on dataset
    ss_sp = [0.12, 0.20] if dname == 'This Work' else [0.08, 0.14]
    chain_sp = mh_sampler(lp_sp, [0.5, 1.5],
                          n_steps=300000, step_size=ss_sp)
    flat_sp  = flat_samples(chain_sp)
    a16,a50,a84 = percentiles(flat_sp[:,1])
    A50 = np.percentile(flat_sp[:,0], 50)
    print(f"    alpha = {a50:.2f} +{a84-a50:.2f} -{a50-a16:.2f}")

    corner_single(flat_sp,
        labels=[r"$\log_{10}\,A$", r"$\alpha$"],
        title=f"Spectral Index — {dname}\n{d['spec_label']}",
        outfile=f"/home/claude/sp_{dname.replace(' ','_')}.pdf",
        color=d['color'],
        xlims=[(-0.5, 3.5), (0.3, 6.0)])

    # ── Shock model ──
    print("  [2] Shock model")
    lp_sh = make_log_prob_shock(d['Fopt'])
    chain_sh = mh_sampler(lp_sh, [43.0, -3.0],
                          n_steps=300000, step_size=[0.55, 0.55])
    flat_sh  = flat_samples(chain_sh)
    Ek95  = np.percentile(flat_sh[:,0], 95)
    N_t   = 200000
    lEk_t = np.random.uniform(42,48,N_t)
    lep_t = np.random.uniform(-5, 0,N_t)
    fex   = np.mean(F_shock_pred(lEk_t,lep_t) > d['Fopt'])
    print(f"    E_k < 10^{Ek95:.1f},  excluded: {fex:.1%}")

    corner_single(flat_sh,
        labels=[r"$\log_{10}(E_k/\mathrm{erg})$",
                r"$\log_{10}\,\varepsilon_B$"],
        title=f"Shock Model — {dname}\n{d['opt_label']}",
        outfile=f"/home/claude/sh_{dname.replace(' ','_')}.pdf",
        color=d['color'],
        xlims=[(42,48),(-5,0)],
        ref_lines=[('v',0,np.log10(E_k_ref),'navy','--',
                    r'Ref $E_k=10^{45}$'),
                   ('h',1,np.log10(epsB_ref),'navy','--','')])

    # ── Magnetospheric efficiencies ──
    print("  [3] Magnetospheric efficiencies")
    eta_o_lim = d['Eopt'] / d['Erad']
    eta_x_lim = d['EX']   / d['Erad']
    lp_mg = make_log_prob_mag(d['Eopt'], d['EX'], d['Erad'])
    chain_mg = mh_sampler(lp_mg, [1.0, 3.0],
                          n_steps=300000, step_size=[0.55, 0.55])
    flat_mg  = flat_samples(chain_mg)
    o95 = np.percentile(flat_mg[:,0], 95)
    x95 = np.percentile(flat_mg[:,1], 95)
    print(f"    eta_opt < 10^{o95:.2f},  eta_X < 10^{x95:.2f}")
    print(f"    SGR analogy: log eta_X = 5.0  "
          f"({'EXCLUDED' if x95 < 5.0 else 'not yet excluded'})")

    corner_single(flat_mg,
        labels=[r"$\log_{10}\,\eta_{\rm opt}$",
                r"$\log_{10}\,\eta_X$"],
        title=f"Magnetospheric Efficiencies — {dname}\n"
              f"Opt: {d['opt_label']}  |  X: {d['X_label']}",
        outfile=f"/home/claude/mg_{dname.replace(' ','_')}.pdf",
        color=d['color'],
        xlims=[(-15,10),(-15,12)],
        ref_lines=[('h',1,5.0,'navy','-.',
                    r'SGR 1935+2154 ($\eta_X\sim10^5$)')])

    results[dname] = {
        'alpha': (a50, a16, a84),
        'A50':   A50,
        'flat_sp': flat_sp, 'flat_sh': flat_sh, 'flat_mg': flat_mg,
        'Ek95': Ek95, 'fex': fex,
        'o95': o95, 'x95': x95,
        'eta_o_lim': eta_o_lim, 'eta_x_lim': eta_x_lim,
        'color': d['color'],
        'Fopt': d['Fopt'],
    }


# ═══════════════════════════════════════════════════════════════
# MASTER COMPARISON FIGURE  (3 × 3 grid)
# ═══════════════════════════════════════════════════════════════
print("\n  Building master comparison figure...")

dnames = ['This Work', 'Literature', 'Combined']
colors = [results[d]['color'] for d in dnames]
labels_sp = [r"$\alpha$  ($F_\nu\propto\nu^{-\alpha}$)"]

fig_master, axes = plt.subplots(3, 3, figsize=(17, 15))
fig_master.suptitle(
    'FRB 20180916B — Broadband Constraints: This Work vs Literature vs Combined',
    fontsize=14, y=0.995)

col_titles = ['This Work\n(SRT/uGMRT + AquEYE+ + LAXPC)',
              'Literature\n(LOFAR/CHIME/Effelsberg + Chandra + Gemini)',
              'Combined\n(All data)']
row_titles = ['Spectral Index  α',
              'Shock Model  $(E_k,\\,\\varepsilon_B)$',
              'Magnetospheric  $(\\eta_{\\rm opt},\\,\\eta_X)$']

for ci, dname in enumerate(dnames):
    r  = results[dname]
    c  = r['color']
    ax = axes[0, ci]

    # ── ROW 0: spectral index 1D marginal ──
    s = r['flat_sp'][:, 1]
    ax.hist(s, bins=60, color=c, alpha=0.50, density=True,
            histtype='stepfilled')
    ax.hist(s, bins=60, color=c, alpha=1.0,  density=True,
            histtype='step', lw=1.5)
    q16, q50, q84 = percentiles(s)
    lo, hi = q50-q16, q84-q50
    ax.axvline(q50, color=c, lw=2.0)
    ax.axvline(q16, color=c, lw=1.0, ls=':')
    ax.axvline(q84, color=c, lw=1.0, ls=':')
    # Bethapudi literature reference
    ax.axvline(2.6, color='gray', lw=1.5, ls='--',
               label=r'Bethapudi+23: $\alpha=2.6$')
    ts = (f"$\\alpha = {q50:.2f}"
          r"^{+" + f"{hi:.2f}" + r"}_{-" + f"{lo:.2f}" + r"}$")
    ax.set_title(ts, fontsize=11)
    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(0.3, 6.0)
    ax.legend(fontsize=8)
    if ci == 0:
        ax.set_ylabel(row_titles[0], fontsize=11)
    ax.text(0.5, 0.95, col_titles[ci], transform=ax.transAxes,
            ha='center', va='top', fontsize=9.5,
            color=c, fontweight='bold')

    # ── ROW 1: shock 2D posterior ──
    ax = axes[1, ci]
    flat_sh = r['flat_sh']
    x, y = flat_sh[:,0], flat_sh[:,1]
    h, xe, ye = np.histogram2d(x, y, bins=45,
                                range=[[42,48],[-5,0]])
    h  = gaussian_filter(h.T, sigma=1.2)
    hf = np.sort(h.ravel())[::-1]
    cm = np.cumsum(hf)/np.sum(hf)
    l68 = hf[np.searchsorted(cm, 0.68)]
    l95 = hf[np.searchsorted(cm, 0.95)]
    X   = 0.5*(xe[:-1]+xe[1:])
    Y   = 0.5*(ye[:-1]+ye[1:])
    ax.contourf(X,Y,h, levels=[l95,l68,h.max()+1],
                colors=[c,c], alpha=[0.20,0.45])
    ax.contour(X,Y,h, levels=[l95,l68], colors=[c], linewidths=1.2)

    # Exclusion boundary
    lEk_g    = np.linspace(42, 48, 300)
    lep_g    = np.linspace(-5, 0,  300)
    LEK, LEP = np.meshgrid(lEk_g, lep_g)
    LOG_F    = np.log10(F_shock_pred(LEK, LEP))
    LOG_LIM  = np.log10(r['Fopt'])
    ax.contour(lEk_g, lep_g,
               LOG_F > LOG_LIM,
               levels=[0.5], colors=['crimson'],
               linewidths=2.0, linestyles='--')
    ax.plot(np.log10(E_k_ref), np.log10(epsB_ref),
            '*', color='navy', ms=12, zorder=10)
    ax.set_xlabel(r'$\log_{10}(E_k/\mathrm{erg})$', fontsize=11)
    ax.set_ylabel(r'$\log_{10}\,\varepsilon_B$', fontsize=11)
    ax.set_xlim(42,48); ax.set_ylim(-5,0)
    ax.text(44.5, -0.4,
            f"excl. {r['fex']:.0%}",
            fontsize=10, ha='center', color='crimson',
            bbox=dict(fc='white', alpha=0.7, ec='crimson'))
    if ci == 0:
        ax.set_ylabel(row_titles[1], fontsize=11)

    # ── ROW 2: magnetospheric 2D posterior ──
    ax = axes[2, ci]
    flat_mg = r['flat_mg']
    x2, y2 = flat_mg[:,0], flat_mg[:,1]
    h2, xe2, ye2 = np.histogram2d(x2, y2, bins=45,
                                   range=[[-15,10],[-15,12]])
    h2  = gaussian_filter(h2.T, sigma=1.2)
    hf2 = np.sort(h2.ravel())[::-1]
    cm2 = np.cumsum(hf2)/np.sum(hf2)
    l682 = hf2[np.searchsorted(cm2, 0.68)]
    l952 = hf2[np.searchsorted(cm2, 0.95)]
    X2   = 0.5*(xe2[:-1]+xe2[1:])
    Y2   = 0.5*(ye2[:-1]+ye2[1:])
    ax.contourf(X2,Y2,h2, levels=[l952,l682,h2.max()+1],
                colors=[c,c], alpha=[0.20,0.45])
    ax.contour(X2,Y2,h2, levels=[l952,l682],
               colors=[c], linewidths=1.2)
    # Limit lines
    ax.axvline(np.log10(r['eta_o_lim']), color='green',
               lw=1.8, ls='--', label=r'$\eta_{\rm opt}$ lim')
    ax.axhline(np.log10(r['eta_x_lim']), color='red',
               lw=1.8, ls='--', label=r'$\eta_X$ lim')
    ax.axhline(5.0, color='navy', lw=1.8, ls='-.',
               label=r'SGR $\eta_X\sim10^5$')
    ax.set_xlabel(r'$\log_{10}\,\eta_{\rm opt}$', fontsize=11)
    ax.set_ylabel(r'$\log_{10}\,\eta_X$', fontsize=11)
    ax.set_xlim(-15,10); ax.set_ylim(-15,12)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.text(0.70, 0.08,
            fr"$\eta_X<10^{{{r['x95']:.1f}}}$",
            transform=ax.transAxes, fontsize=10,
            color='darkred',
            bbox=dict(fc='white', alpha=0.7, ec='red'))
    if ci == 0:
        ax.set_ylabel(row_titles[2], fontsize=11)

plt.tight_layout(rect=[0,0,1,0.985])
plt.savefig('/home/claude/fig_master_comparison.pdf',
            dpi=180, bbox_inches='tight')
plt.close()
print("    Saved: fig_master_comparison.pdf")


# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("COMPARISON SUMMARY")
print(f"{'Quantity':<30} {'This Work':>12} {'Literature':>12} {'Combined':>12}")
print("-" * 65)

for dname in dnames:
    pass  # printed below

a_tw  = results['This Work']['alpha']
a_lit = results['Literature']['alpha']
a_co  = results['Combined']['alpha']
print(f"{'alpha (median)':<30} "
      f"{a_tw[0]:>12.2f} {a_lit[0]:>12.2f} {a_co[0]:>12.2f}")

print(f"{'Shock excl. (prior vol.)':<30} "
      f"{results['This Work']['fex']:>12.1%} "
      f"{results['Literature']['fex']:>12.1%} "
      f"{results['Combined']['fex']:>12.1%}")

print(f"{'E_k 95th pct (log erg)':<30} "
      f"{results['This Work']['Ek95']:>12.1f} "
      f"{results['Literature']['Ek95']:>12.1f} "
      f"{results['Combined']['Ek95']:>12.1f}")

print(f"{'eta_X 95th pct (log)':<30} "
      f"{results['This Work']['x95']:>12.2f} "
      f"{results['Literature']['x95']:>12.2f} "
      f"{results['Combined']['x95']:>12.2f}")

print(f"{'SGR analogy excluded?':<30} "
      f"{'Yes' if results['This Work']['x95']   < 5.0 else 'No':>12} "
      f"{'Yes' if results['Literature']['x95']  < 5.0 else 'No':>12} "
      f"{'Yes' if results['Combined']['x95']    < 5.0 else 'No':>12}")
print("=" * 65)
print("\nOutput files:")
for dname in dnames:
    dn = dname.replace(' ','_')
    for prefix in ['sp','sh','mg']:
        print(f"  {prefix}_{dn}.pdf")
print("  fig_master_comparison.pdf  ← main thesis figure")
