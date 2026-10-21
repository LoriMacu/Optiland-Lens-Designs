"""
Summary of the optimised 0.5x object-telecentric objective:
first-order data, image quality, third-order (Seidel) aberration sums and a
Zernike breakdown of the real wavefront at every field.

Usage:  python Telecentric_summary.py telecentric_best.json
"""

import sys

import numpy as np
import pandas as pd
from optiland.fileio import load_optiland_file
from optiland.wavefront import OPD

import Telecentric_design as T

path = sys.argv[1] if len(sys.argv) > 1 else "telecentric_best.json"
lens = load_optiland_file(path)
v = T.system_values(lens)
n = lens.surfaces.num_surfaces
z = np.asarray(lens.surfaces.positions).ravel()
semi = v["semi"]
spots = T.spot_sizes(lens, T.RINGS_FINAL) * 1e3
wfe = T.rms_wavefront(lens)
strehl = np.exp(-(2 * np.pi * wfe) ** 2)
glasses = [lens.surfaces[a].material_post.name for a, _ in T.elements(lens)]
cemented = sum(1 for (a1, b1), (a2, _) in zip(T.elements(lens), T.elements(lens)[1:]) if b1 == a2)
airy = 1.22 * T.PRIMARY_WL * v["working_fno"]

# ---------------------------------------------------------------- characteristics
rows = [
    ("Magnification", f"{v['mag']:.4f}x"),
    ("Object-space NA / image-space NA", f"{T.OBJECT_NA:.3f} / {T.OBJECT_NA / abs(v['mag']):.3f}"),
    ("Working F/#", f"{v['working_fno']:.2f}"),
    ("Working distance (object to S1)", f"{T.WD:.1f} mm"),
    ("Lens length (S1 to image)", f"{v['length']:.2f} mm"),
    ("Total track (object to image)", f"{z[-1] - z[0]:.2f} mm"),
    ("Back focal distance", f"{v['bfd']:.2f} mm"),
    ("Object field (diameter)", f"{2 * T.OBJ_HEIGHTS[-1]:.1f} mm"),
    ("Image circle (diameter)", f"{2 * T.OBJ_HEIGHTS[-1] * abs(v['mag']):.1f} mm (2/3\" sensor)"),
    ("Elements / groups", f"{v['n_elements']} / {v['n_elements'] - cemented}"),
    ("Glasses", ", ".join(dict.fromkeys(glasses))),
    ("Largest / smallest clear diameter", f"{2 * max(semi):.2f} / {2 * min(semi):.2f} mm"),
    ("Stop diameter", f"{2 * semi[lens.surfaces.stop_index - 1]:.2f} mm"),
    ("Object-space telecentricity (max)", f"{max(abs(a) for a in v['tele']):.4f} deg"),
    ("Image-space CRA (max)", f"{max(abs(a) for a in v['cra']):.2f} deg"),
    ("Distortion (max)", f"{max(abs(d) for d in v['dist']):.3f} %"),
    ("Polychromatic RMS spot (mean / max)", f"{spots.mean():.2f} / {spots.max():.2f} um"),
    ("Airy radius (588 nm)", f"{airy:.2f} um"),
    ("RMS wavefront error (worst)", f"{wfe.max():.3f} waves"),
    ("Strehl (worst, Marechal)", f"{strehl.min():.2f}"),
    ("Depth of field (object, +-lambda/2NA^2)", f"+-{T.PRIMARY_WL * 1e-3 / (2 * T.OBJECT_NA ** 2):.2f} mm"),
    ("Optical resolution (object, Rayleigh)", f"{0.61 * T.PRIMARY_WL / T.OBJECT_NA:.1f} um"),
]
print("\nLENS CHARACTERISTICS")
print(pd.DataFrame(rows, columns=["Parameter", "Value"]).to_string(index=False))

# ---------------------------------------------------------------- third order
ab = lens.aberrations
third = {
    "Spherical (TSC)": ab.TSC(), "Coma (TCC)": ab.TCC(), "Astigmatism (TAC)": ab.TAC(),
    "Petzval (TPC)": ab.TPC(), "Distortion (DC)": ab.DC(),
}
# (Optiland's first-order color sums TAchC/LchC do not match real rays for this
#  finite-conjugate system, so color is measured with real rays further below)
third = pd.DataFrame({k: np.asarray(val, dtype=float) * 1e3 for k, val in third.items()})
tot = third.sum()
largest = third.abs().max()
print("\nTHIRD-ORDER TRANSVERSE ABERRATIONS at the image, full field / full aperture [um]")
print(pd.DataFrame({"System total": tot.round(2), "Largest single surface": largest.round(1),
                    "Total / largest [%]": (100 * tot.abs() / largest).round(1)}).to_string())


# ---------------------------------------------------------------- Zernike breakdown
def fringe_terms(r, t):
    return {
        "Defocus": [2 * r ** 2 - 1],
        "Astigmatism": [r ** 2 * np.cos(2 * t), r ** 2 * np.sin(2 * t)],
        "Coma": [(3 * r ** 3 - 2 * r) * np.cos(t), (3 * r ** 3 - 2 * r) * np.sin(t)],
        "Spherical": [6 * r ** 4 - 6 * r ** 2 + 1],
        "Trefoil": [r ** 3 * np.cos(3 * t), r ** 3 * np.sin(3 * t)],
        "2nd astigmatism": [(4 * r ** 4 - 3 * r ** 2) * np.cos(2 * t), (4 * r ** 4 - 3 * r ** 2) * np.sin(2 * t)],
        "2nd coma": [(10 * r ** 5 - 12 * r ** 3 + 3 * r) * np.cos(t), (10 * r ** 5 - 12 * r ** 3 + 3 * r) * np.sin(t)],
        "2nd spherical": [20 * r ** 6 - 30 * r ** 4 + 12 * r ** 2 - 1,
                          70 * r ** 8 - 140 * r ** 6 + 90 * r ** 4 - 20 * r ** 2 + 1],
    }


def zernike_breakdown(h, wl):
    """RMS (waves) of each aberration family in the real wavefront (tilt/piston removed)."""
    opd = OPD(lens, (0.0, h / T.OBJ_HEIGHTS[-1]), wl, num_rays=16, strategy="centroid")
    data = opd.get_data(opd.fields[0], opd.wavelengths[0])
    x, y = np.asarray(opd.distribution.x), np.asarray(opd.distribution.y)
    w, keep = np.asarray(data.opd), np.asarray(data.intensity) > 0
    x, y, w = x[keep], y[keep], w[keep]
    r, t = np.hypot(x, y), np.arctan2(y, x)
    fam = fringe_terms(r, t)
    cols = [np.ones_like(r), r * np.cos(t), r * np.sin(t)] + [c for v_ in fam.values() for c in v_]
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    out, j = {}, 3
    for name, v_ in fam.items():
        part = A[:, j:j + len(v_)] @ coef[j:j + len(v_)]
        out[name] = np.std(part)
        j += len(v_)
    resid = w - A @ coef
    out["Higher order"] = np.std(resid)
    out["Total RMS"] = np.std(w - A[:, 1:3] @ coef[1:3])
    return out


print("\nREAL WAVEFRONT, Zernike breakdown at 588 nm [RMS waves]  (diffraction limit: total <= 0.071)")
zt = pd.DataFrame({f"h={h:g} mm": zernike_breakdown(h, T.PRIMARY_WL) for h in T.OBJ_HEIGHTS})
print(zt.round(4).to_string())
dom = zt.drop("Total RMS").idxmax()
print("Dominant term per field:", ", ".join(f"{c}: {d}" for c, d in dom.items()))

print("\nREAL WAVEFRONT, worst field (h = 11 mm) per wavelength [RMS waves]")
zw = pd.DataFrame({f"{wl * 1e3:.0f} nm": zernike_breakdown(T.OBJ_HEIGHTS[-1], wl) for wl in T.WAVELENGTHS})
print(zw.round(4).to_string())

# ---------------------------------------------------------------- chromatic (real rays)
def ray(hy, py, wl):
    lens.trace_generic(np.zeros(1), np.array([hy]), np.zeros(1), np.array([py]), wl)
    s_ = lens.surfaces
    return (float(np.asarray(s_.y)[-1][0]), float(np.asarray(s_.M)[-1][0]), float(np.asarray(s_.N)[-1][0]))


focus = {}
for wl in T.WAVELENGTHS:
    y, M, N = ray(0.0, 0.707, wl)                     # 0.707 zone, on axis
    focus[wl] = -y * N / M * 1e3                       # axial crossing vs sensor [um]
y_chief = {wl: ray(1.0, 0.0, wl)[0] for wl in T.WAVELENGTHS}
b, g, r = T.WAVELENGTHS
dof_img = 2 * T.PRIMARY_WL * v["working_fno"] ** 2
print("\nCHROMATIC ABERRATIONS (real rays)")
print(pd.DataFrame([
    ("Axial color, focus 486 - 656 nm", f"{focus[b] - focus[r]:.1f} um"),
    ("Secondary spectrum, focus 588 - mean(486, 656)", f"{focus[g] - (focus[b] + focus[r]) / 2:.1f} um"),
    ("Image-side depth of focus", f"+-{dof_img:.0f} um"),
    ("Lateral color at full field, 486 - 656 nm", f"{(y_chief[b] - y_chief[r]) * 1e3:.2f} um"),
], columns=["Quantity", "Value"]).to_string(index=False))
