# FRB 20180916B — Broadband MCMC Analysis

> **Chapter 6 analysis pipeline** for the thesis *Broadband Constraints on FRB 20180916B*.  
> Performs Metropolis-Hastings MCMC inference on three datasets across three physical analyses, producing corner plots and a master comparison figure.

---

## Overview

This script implements a full Bayesian characterisation of FRB 20180916B using data from:

| Dataset | Instruments / Sources |
|---|---|
| **This Work** | SRT (328 MHz), uGMRT, AquEYE+ (optical), LAXPC (X-ray) |
| **Literature** | LOFAR, CHIME, Effelsberg, Chandra, Gemini/'Alopeke |
| **Combined** | All of the above, using the most constraining limit per band |

For each dataset, three analyses are run:

1. **Spectral index** α — fits F_ν ∝ ν^{−α} to detections with soft upper-limit penalties
2. **Shock model** — constrains (log E_k, log ε_B) via the optical non-detection
3. **Magnetospheric efficiencies** — constrains (log η_opt, log η_X) relative to the radio burst energy

---

## Requirements

```bash
pip install numpy matplotlib scipy
```

| Package | Tested version |
|---|---|
| `numpy` | ≥ 1.24 |
| `matplotlib` | ≥ 3.7 |
| `scipy` | ≥ 1.10 |

Python 3.9+ recommended. No GPU required.

---

## Usage

```bash
python mcmc_frb_20180916B_ch6.py
```

The script is self-contained. All data values are defined inline with references to the thesis tables and external papers.

**Output files** written to the working directory:

| File | Contents |
|---|---|
| `sp_This_Work.pdf` | Corner plot: spectral index, This Work dataset |
| `sp_Literature.pdf` | Corner plot: spectral index, Literature dataset |
| `sp_Combined.pdf` | Corner plot: spectral index, Combined dataset |
| `sh_This_Work.pdf` | Corner plot: shock model (E_k, ε_B), This Work |
| `sh_Literature.pdf` | Corner plot: shock model, Literature |
| `sh_Combined.pdf` | Corner plot: shock model, Combined |
| `mg_This_Work.pdf` | Corner plot: magnetospheric efficiencies (η_opt, η_X), This Work |
| `mg_Literature.pdf` | Corner plot: magnetospheric efficiencies, Literature |
| `mg_Combined.pdf` | Corner plot: magnetospheric efficiencies, Combined |
| `fig_master_comparison.pdf` | **Main thesis figure** — 3×3 grid comparing all datasets and analyses |

---

## Method

### Sampler

Metropolis-Hastings with hand-tuned Gaussian proposal distributions. Step sizes are set per-dataset to achieve acceptance rates of 20–40% (flagged at runtime if outside 15–55%). Each chain runs for 300,000 steps; the first 40% is discarded as burn-in and the remainder thinned by a factor of 25.

```
chain length : 300,000 steps
burn-in      : 40%  (120,000 steps)
thinning     : every 25th sample
flat samples : ~4,800 per parameter
```

### Likelihood functions

**Spectral index** — Gaussian likelihood on log-flux detections plus a complementary error function (`erfc`) penalty for each upper limit, softened by σ_UL = 0.15 dex:

```
log L = −½ Σ [(log F_obs − log F_model)² / σ²]
      + Σ  log[½ erfc((log F_model − log F_lim) / (√2 σ_UL))]
```

**Shock model** — flat prior in the allowed region, Gaussian penalty when the predicted optical flux exceeds the non-detection limit (σ = 0.20 dex). The predicted flux scales as:

```
F_shock ∝ (E_k / E_k,ref)^1.05 × (ε_B / ε_B,ref)^0.10
```

**Magnetospheric efficiencies** — flat prior, Gaussian penalty when either η_opt = E_opt/E_radio or η_X = E_X/E_radio exceeds its respective upper-limit (σ = 0.15 dex).

### Priors

| Parameter | Range |
|---|---|
| log₁₀ A (spectral norm) | (−0.5, 3.5) |
| α (spectral index) | (0.3, 6.0) |
| log₁₀ E_k / erg | (42, 48) |
| log₁₀ ε_B | (−5, 0) |
| log₁₀ η_opt | (−15, 12) |
| log₁₀ η_X | (−15, 14) |

---

## Key Observational Inputs

All values are cited with their source in the script comments.

| Quantity | Value | Source |
|---|---|---|
| SRT detections (328 MHz) | 1.9–6.2 Jy ms | Table 3.19 |
| SRT upper limits | 0.30–4.00 Jy ms | Table 3.14 |
| AquEYE+ 5σ 1ms flux limit | 3.1 × 10⁻¹⁵ erg/cm² | Chapter 4 |
| LAXPC 3σ 1ms flux limit | 3.76 × 10⁻¹¹ erg/cm² | Chapter 5 |
| Chandra flux limit | 5.0 × 10⁻¹⁰ erg/cm² | Scholz et al. 2020 |
| Gemini/'Alopeke flux limit | 1.0 × 10⁻¹² erg/cm² | Kilpatrick et al. 2023 |
| Luminosity distance | 147 Mpc | — |

---

## Output Interpretation

### Spectral index
The marginal posterior on α is compared against the Bethapudi et al. (2023) reference value of α = 2.6 (shown as a dashed vertical line). Steep values (α ≳ 3) are consistent with a sharp spectral turnover or scattering.

### Shock model
The crimson dashed contour marks the observational exclusion boundary in (log E_k, log ε_B) space — parameter combinations above this line predict optical fluxes exceeding the non-detection limit. The navy star marks the reference point (E_k = 10⁴⁵ erg, ε_B = 0.01). The fraction of prior volume excluded is printed on each panel.

### Magnetospheric efficiencies
Green and red dashed lines mark the 95th-percentile upper limits on η_opt and η_X respectively. The navy dash-dot line at log η_X = 5 corresponds to the SGR 1935+2154 analogy (η_X ~ 10⁵); whether this is excluded depends on the dataset.

---

## References

- Pleunis et al. (2021) — LOFAR detections  
- CHIME/FRB Collaboration (2019, 2021) — CHIME detections  
- Bethapudi et al. (2023) — Effelsberg detections, spectral index  
- Scholz et al. (2020) — Chandra X-ray upper limits  
- Kilpatrick et al. (2023) — Gemini/'Alopeke optical upper limits  
- Gopinath et al. (2024) — LOFAR activity window  

---

## Notes

- The random seed is fixed (`np.random.seed(42)`) for reproducibility.
- Acceptance rates outside the 15–55% range trigger a `⚠ CHECK` warning; step sizes can be adjusted in the `mh_sampler` call for each analysis block.
- The `Combined` dataset uses `AquEYE+` as the optical limit (deeper than Gemini by ~3 orders of magnitude) and `Chandra` as the X-ray limit (more constraining than LAXPC in the soft band).
