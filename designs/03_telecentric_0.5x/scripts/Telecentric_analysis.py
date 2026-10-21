"""
Analysis of the optimised 0.5x object-telecentric objective:
layout, spot diagram, polychromatic MTF, telecentricity / distortion vs field,
and MTF at the Nyquist frequency of real 2/3" machine-vision sensors.

Usage:  python Telecentric_analysis.py telecentric_best.json
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optiland.analysis import SpotDiagram
from optiland.fileio import load_optiland_file
from optiland.mtf import ScalarFFTMTF

import Telecentric_design as T

# Typical 2/3"-class machine-vision sensors (pixel pitch in um)
SENSORS = [
    ("Sony IMX250 / IMX264 (5 MP, 2/3\")", 3.45),
    ("Generic 4.5 um pixel", 4.5),
    ("Generic 5.5 um pixel", 5.5),
    ("Sony ICX285 (1.4 MP CCD, 2/3\")", 6.45),
]

path = sys.argv[1] if len(sys.argv) > 1 else "telecentric_best.json"
prefix = path.replace(".json", "")
lens = load_optiland_file(path)
T.report(lens)

# ---------------------------------------------------------------- layout
lens.draw(num_rays=5, figsize=(14, 4))
plt.title("0.5x object-space telecentric objective, WD 110 mm, NA 0.028")
plt.savefig(f"{prefix}_layout.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- spot diagram
spot = SpotDiagram(lens, num_rings=16, reference="centroid")
spot.view()
plt.savefig(f"{prefix}_spots.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- polychromatic MTF
# Optiland's FFT MTF is monochromatic: compute it at each wavelength and average
# the curves (equal weights) on a common frequency axis.
v = T.system_values(lens)
fno = v["working_fno"]
f_max = 1 / (min(T.WAVELENGTHS) * 1e-3 * fno)          # highest cutoff (blue)
freq = np.linspace(0, f_max, 400)
n_fields = len(T.OBJ_HEIGHTS)
tang = np.zeros((n_fields, freq.size))
sag = np.zeros((n_fields, freq.size))
diff_lim = np.zeros(freq.size)
for wl in T.WAVELENGTHS:
    mtf = ScalarFFTMTF(lens, fields="all", wavelength=wl, num_rays=128, grid_size=512)
    for k in range(n_fields):
        tang[k] += np.interp(freq, np.asarray(mtf.freq_tang[k]), np.asarray(mtf.mtf[k][0]), right=0) / 3
        sag[k] += np.interp(freq, np.asarray(mtf.freq_sag[k]), np.asarray(mtf.mtf[k][1]), right=0) / 3
    u = np.clip(freq * wl * 1e-3 * fno, 0, 1)
    diff_lim += (2 / np.pi) * (np.arccos(u) - u * np.sqrt(1 - u ** 2)) / 3

fig, ax = plt.subplots(figsize=(11, 6))
for k, h in enumerate(T.OBJ_HEIGHTS):
    ax.plot(freq, tang[k], color=f"C{k}", label=f"object h = {h:.1f} mm  T")
    ax.plot(freq, sag[k], color=f"C{k}", ls="--", label=f"object h = {h:.1f} mm  S")
ax.plot(freq, diff_lim, "k", lw=2, label="Diffraction limit")
for j, (name, p) in enumerate(SENSORS):
    ny = 1 / (2 * p * 1e-3)
    ax.axvline(ny, color="0.5", ls=":", lw=1)
    ax.text(ny, 0.97 - 0.09 * (j % 2),f" Nyquist {p} um\n {ny:.0f} lp/mm", fontsize=8, va="top", color="0.3")
ax.set_xlim(0, 200)
ax.set_ylim(0, 1)
ax.set_xlabel("Image-space frequency [lp/mm]")
ax.set_ylabel("Polychromatic MTF")
ax.set_title(f"Polychromatic MTF (486/588/656 nm), working f/{fno:.1f}")
ax.grid(alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{prefix}_mtf.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- telecentricity & distortion
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(T.OBJ_HEIGHTS, v["tele"], "o-")
a1.axhspan(-T.TELE_MAX, T.TELE_MAX, color="tab:green", alpha=0.15, label="spec +-0.10 deg")
a1.set_xlabel("Object height [mm]"); a1.set_ylabel("Object-space chief-ray angle [deg]")
a1.set_title("Telecentricity"); a1.legend(); a1.grid(alpha=0.3)
a2.plot(T.OBJ_HEIGHTS, v["dist"], "o-", color="tab:orange")
a2.axhspan(-T.DIST_MAX, T.DIST_MAX, color="tab:green", alpha=0.15, label="spec +-0.10 %")
a2.set_xlabel("Object height [mm]"); a2.set_ylabel("Distortion [%]")
a2.set_title("Distortion"); a2.legend(); a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{prefix}_tele_dist.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- sensor match
rows = []
for name, p in SENSORS:
    ny = 1 / (2 * p * 1e-3)
    worst = min(min(np.interp(ny, freq, tang[k]), np.interp(ny, freq, sag[k])) for k in range(n_fields))
    rows.append({"Sensor": name, "Pixel [um]": p, "Nyquist [lp/mm]": round(ny, 1),
                 "Object-side pixel [um]": round(p / abs(v["mag"]), 2),
                 "MTF diffraction limit": round(float(np.interp(ny, freq, diff_lim)), 3),
                 "MTF worst field (T/S)": round(float(worst), 3),
                 "MTF @ Nyquist/2 worst": round(float(min(min(np.interp(ny / 2, freq, tang[k]),
                                                              np.interp(ny / 2, freq, sag[k]))
                                                          for k in range(n_fields))), 3)})
print("\nMTF at the sensor Nyquist frequency")
print(pd.DataFrame(rows).to_string(index=False))
