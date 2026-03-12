\begin{lstlisting}[language=Python, caption={Python script for chromatic modelling of FRB~20180916B and overlay of July 2024 SRT sessions.}, label={lst:chromatic_code}]
"""
Chromatic modelling of FRB 20180916B with July 2024 SRT sessions (04:00–12:00 UT)
Model based on Bethapudi et al. (2022, MNRAS preprint 2207.13669)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Model parameters (Bethapudi+ 2022)
# -----------------------------------------------------------
phi_B, phi_A = 0.47, -0.23            # peak phase power-law coefficients
fwhm_B_hours, fwhm_A = 53.4, -0.35    # FWHM power-law coefficients
P_days = 16.33                        # activity period in days (ZP21)
P_hours = P_days * 24.0
MJD_ref = 58369.40                    # reference epoch (ZP21)

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def phi_peak_from_freq_GHz(f_GHz):
    """Return activity peak phase for given observing frequency (GHz)."""
    f_MHz = f_GHz * 1000.0
    return phi_B * (f_MHz / 600.0) ** phi_A

def fwhm_phase_from_freq_GHz(f_GHz):
    """Return full width at half maximum (FWHM) in phase units."""
    f_MHz = f_GHz * 1000.0
    fwhm_hours = fwhm_B_hours * (f_MHz / 600.0) ** fwhm_A
    return fwhm_hours / P_hours

def phase_of_mjd(mjd):
    """Convert MJD to activity phase using ZP21 ephemeris."""
    return ((mjd - MJD_ref) / P_days) % 1.0

# -----------------------------------------------------------
# Define SRT sessions (04:00–12:00 UT each day)
# -----------------------------------------------------------
sessions = pd.DataFrame([
    {"Date": "2024-07-21", "Band": "K",      "f_GHz": 22.0, "MJD_day": 60593.0},
    {"Date": "2024-07-22", "Band": "C-high", "f_GHz": 6.7,  "MJD_day": 60594.0},
    {"Date": "2024-07-23", "Band": "C-low",  "f_GHz": 5.0,  "MJD_day": 60595.0},
])

# Convert to MJD start/end for 04:00–12:00 UT
sessions["MJD_start"] = sessions["MJD_day"] + 4 / 24.0
sessions["MJD_end"]   = sessions["MJD_day"] + 12 / 24.0
sessions["phi_start"] = sessions["MJD_start"].apply(phase_of_mjd)
sessions["phi_end"]   = sessions["MJD_end"].apply(phase_of_mjd)

# -----------------------------------------------------------
# Compute chromatic windows
# -----------------------------------------------------------
def window_for_band(f_GHz):
    """Return activity window center and limits for a given frequency."""
    phi_c = phi_peak_from_freq_GHz(f_GHz)
    halfw = 0.5 * fwhm_phase_from_freq_GHz(f_GHz)
    return phi_c, halfw, (phi_c - halfw) % 1.0, (phi_c + halfw) % 1.0

windows = []
for _, r in sessions.iterrows():
    phi_c, halfw, w0, w1 = window_for_band(r["f_GHz"])
    windows.append({
        "Band": r["Band"], "f_GHz": r["f_GHz"],
        "phi_center": phi_c, "half_width": halfw,
        "win_start": w0, "win_end": w1
    })
wins = pd.DataFrame(windows)

# -----------------------------------------------------------
# Compute overlap between observation and window
# -----------------------------------------------------------
def overlap_category(phi_s, phi_e, w0, w1):
    """Determine if observation overlaps with predicted window."""
    def segs(a,b):
        if b >= a: return [(a,b)]
        else: return [(a,1.0),(0.0,b)]
    overlap = 0.0
    for a0,a1 in segs(phi_s,phi_e):
        for b0,b1 in segs(w0,w1):
            lo, hi = max(a0,b0), min(a1,b1)
            if hi > lo: overlap += (hi - lo)
    if overlap <= 1e-6: return "No"
    obs_len = (phi_e - phi_s) if phi_e >= phi_s else (1.0 - phi_s + phi_e)
    if abs(overlap - obs_len) < 1e-6: return "Yes (full)"
    return "Partial"

summary = []
for i, r in sessions.iterrows():
    w = wins.iloc[i]
    cat = overlap_category(r["phi_start"], r["phi_end"], w["win_start"], w["win_end"])
    summary.append({
        "Date": r["Date"],
        "Band": r["Band"],
        "f_GHz": r["f_GHz"],
        "MJD start": r["MJD_start"],
        "MJD end": r["MJD_end"],
        "Phase start": r["phi_start"],
        "Phase end": r["phi_end"],
        "Window center": w["phi_center"],
        "Window start": w["win_start"],
        "Window end": w["win_end"],
        "In window?": cat
    })
summary_df = pd.DataFrame(summary)

print("\n=== SRT sessions vs. chromatic windows ===")
print(summary_df[["Band","Date","f_GHz","Phase start","Phase end",
                  "Window start","Window end","In window?"]])

# -----------------------------------------------------------
# Plot phase–frequency relation
# -----------------------------------------------------------
freqs = np.logspace(np.log10(0.1), np.log10(26.0), 300)
phi_centers = np.array([phi_peak_from_freq_GHz(f) for f in freqs])
phi_halfwidths = 0.5 * np.array([fwhm_phase_from_freq_GHz(f) for f in freqs])

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(phi_centers, freqs, "k-", lw=2, label="Peak phase")
ax.fill_betweenx(freqs, phi_centers - phi_halfwidths, phi_centers + phi_halfwidths,
                 alpha=0.4, color="gray", label="Activity window")

# Add SRT observing bars
for _, r in sessions.iterrows():
    ax.hlines(r["f_GHz"], r["phi_start"], r["phi_end"], colors="black", lw=3)
    ax.text((r["phi_start"]+r["phi_end"])/2.0, r["f_GHz"]*1.05,
            f"{r['Band']} ({r['Date']})", ha="center", va="bottom", fontsize=8)

ax.set_yscale("log")
ax.set_xlim(0,1)
ax.set_ylim(0.1,30.0)
ax.set_xlabel("Activity phase")
ax.set_ylabel("Frequency [GHz]")
ax.set_title("Chromatic modelling of FRB 20180916B with SRT (04:00–12:00 UT)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()

plt.savefig("srt_chromatic_model_0400_1200.png", dpi=300)
plt.show()
\end{lstlisting}

\section{Output and Interpretation}

The script generates a console table showing whether each SRT session overlaps 
with the predicted activity window, and produces a plot (\autoref{fig:chromatic_model})
displaying the frequency–phase relation of FRB~20180916B.

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.8\textwidth]{appendix_figs/srt_chromatic_model_0400_1200.png}
    \caption{Chromatic modelling of FRB~20180916B showing the predicted activity window 
    (grey band) and the July~2024 SRT sessions (black bars). The active phase shifts 
    toward earlier values at higher frequencies.}
    \label{fig:chromatic_model}
\end{figure}

\section*{Software Environment}
\begin{itemize}
    \item Python~3.11.5
    \item \texttt{numpy}~1.26, \texttt{pandas}~2.2, \texttt{matplotlib}~3.8
\end{itemize}

\bigskip
The chromatic model confirms that all three SRT sessions fall within or close to 
the predicted active windows, validating the scheduling strategy adopted for the 
July 2024 observation session
