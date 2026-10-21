"""
Image simulation of the 0.5x object-telecentric objective on a real
machine-vision sensor (Zemax "Image Simulation" style).

Pipeline (one pass per wavelength 486 / 588 / 656 nm, equal weights = white LED):
  1. The object scene (a synthetic inspection target, defined in mm) is rendered
     directly on the image plane through the real chief-ray mapping, so
     magnification, distortion and lateral color are exact (real rays).
  2. Spatially variable blur: diffraction PSFs (Optiland FFT PSF) are computed at
     several field radii, rotated to each node of a grid over the sensor and
     blended bilinearly between nodes.
  3. Relative illumination from the real object-space NA vs field.
  4. Sensor: 3.45 um pixels (IMX264-class 2448 x 2048, 2/3"), pixel integration,
     shot + read noise, 8-bit output.
A perfect lens with the same NA (Airy PSF, no distortion) and the ideal
(unblurred) pixel image are simulated for comparison.

Usage:  python Telecentric_image_sim.py telecentric_best.json
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from optiland.fileio import load_optiland_file
from optiland.psf import FFTPSF
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.special import j1

import Telecentric_design as T

path = sys.argv[1] if len(sys.argv) > 1 else "telecentric_best.json"
prefix = path.replace(".json", "")
lens = load_optiland_file(path)

# ---------------------------------------------------------------- sensor and sampling
PIX = 3.45e-3                    # sensor pixel [mm]
NX, NY = 2448, 2048              # IMX264-class, 8.45 x 7.07 mm (2/3")
OS = 2                           # simulation oversampling per pixel
S = PIX / OS                     # simulation sample on the image [mm]
KH = 24                          # PSF kernel half size [samples] (+-41 um)
FULL_WELL, READ_NOISE, EXPOSURE = 10000.0, 2.4, 0.95   # e-, e-, fraction of full well at reflectance 1
WLS = T.WAVELENGTHS
u = (np.arange(NX * OS) - (NX * OS - 1) / 2) * S        # image x of sample centres [mm]
w_ = (np.arange(NY * OS) - (NY * OS - 1) / 2) * S       # image y (row index increases upwards)
R_MAX = np.hypot(u[-1], w_[-1])

# ---------------------------------------------------------------- real-ray mapping
# image radius vs object radius per wavelength (chief rays); shown with the
# 180 deg inversion removed, i.e. as the camera software would display it
r_obj = np.linspace(0, T.OBJ_HEIGHTS[-1], 120)
Hy = r_obj / T.OBJ_HEIGHTS[-1]
rho = {}
for wl in WLS:
    lens.trace_generic(np.zeros_like(Hy), Hy, np.zeros_like(Hy), np.zeros_like(Hy), wl)
    rho[wl] = np.abs(np.asarray(lens.surfaces.y)[-1])
    # the sensor corner is 0.1 % beyond the design field: extend linearly
    slope = (rho[wl][-1] - rho[wl][-2]) / (r_obj[-1] - r_obj[-2])
    rho[wl] = np.append(rho[wl], rho[wl][-1] + slope * 0.02 * T.OBJ_HEIGHTS[-1])
r_obj = np.append(r_obj, 1.02 * T.OBJ_HEIGHTS[-1])
m0 = float(np.polyfit(r_obj[1:8], rho[T.PRIMARY_WL][1:8], 1)[0])        # near-axis |m|
print(f"|m| = {m0:.5f}, sensor corner at object radius {R_MAX / m0:.2f} mm")

# relative illumination: object-space NA (etendue) per field, from 4 marginal rays
ri_h = np.array(T.OBJ_HEIGHTS)
na2 = []
for h in ri_h:
    s2 = []
    for px, py in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        lens.trace_generic(np.zeros(1), np.array([h / T.OBJ_HEIGHTS[-1]]), np.array([px * 1.0]),
                           np.array([py * 1.0]), T.PRIMARY_WL)
        L0, M0, N0 = (float(np.asarray(a)[0][0]) for a in (lens.surfaces.L, lens.surfaces.M, lens.surfaces.N))
        lens.trace_generic(np.zeros(1), np.array([h / T.OBJ_HEIGHTS[-1]]), np.zeros(1), np.zeros(1), T.PRIMARY_WL)
        L1, M1, N1 = (float(np.asarray(a)[0][0]) for a in (lens.surfaces.L, lens.surfaces.M, lens.surfaces.N))
        s2.append(1 - (L0 * L1 + M0 * M1 + N0 * N1) ** 2)
    na2.append(np.mean(s2))
ri = np.array(na2) / na2[0]
print("Relative illumination at object h =", ri_h, ":", np.round(ri, 4))


def image_to_object(U, V, wl):
    """Object coordinates [mm] of image points (U, V) for wavelength wl (real mapping)."""
    rr = np.hypot(U, V)
    ro = np.interp(rr, rho[wl], r_obj)
    k = np.divide(ro, rr, out=np.full_like(rr, 1 / m0), where=rr > 0)
    return U * k, V * k


# ---------------------------------------------------------------- object scene
BG = 0.85
features = []        # (kind, params, value, bbox (x0, x1, y0, y1) in object mm)


def add_rect(cx, cy, w, h, val, ang=0.0):
    e = 0.5 * (abs(w * np.cos(ang)) + abs(h * np.sin(ang))) + 0.01
    f = 0.5 * (abs(w * np.sin(ang)) + abs(h * np.cos(ang))) + 0.01
    features.append(("rect", (cx, cy, w, h, ang), val, (cx - e, cx + e, cy - f, cy + f)))


def add_circle(cx, cy, r, val):
    features.append(("circle", (cx, cy, r), val, (cx - r - 0.01, cx + r + 0.01, cy - r - 0.01, cy + r + 0.01)))


BAR_W = [5, 6.5, 8, 10, 13, 16, 20, 25]          # bar width [um] (period = 2 x width)
PIT_D = [3, 5, 8, 10, 15, 20, 30, 40]            # pit diameter [um]
SCR_W = [1.5, 2, 3, 5, 8]                        # scratch width [um]


def test_cell(x0, y0):
    """Inspection test cell (~1.0 x 0.9 mm) centred at (x0, y0) object mm."""
    x = x0 - 0.48
    for wb in BAR_W:                              # vertical bars (horizontal resolution)
        w = wb * 1e-3
        L = max(0.08, 8 * w)
        for k in range(3):
            add_rect(x + w / 2 + 2 * k * w, y0 + 0.30, w, L, 0.05)
        x += 5 * w + 0.06
    x = x0 - 0.48
    for wb in BAR_W:                              # horizontal bars (vertical resolution)
        w = wb * 1e-3
        L = max(0.06, 6 * w)
        for k in range(3):
            add_rect(x + L / 2, y0 + 0.08 - 2 * w + 2 * k * w, L, w, 0.05)
        x += L + 0.03
    for i, d in enumerate(PIT_D):                 # pits
        add_circle(x0 - 0.42 + 0.115 * i, y0 - 0.14, d * 1e-3 / 2, 0.35)
    for i, wb in enumerate(SCR_W):                # scratches at 30 deg
        add_rect(x0 - 0.36 + 0.18 * i, y0 - 0.33, 0.25, wb * 1e-3, 0.45, np.radians(30))


CELLS = {"Centre": (0.0, 0.0), "Mid field": (-4.2, 3.2), "Corner": (7.8, -6.4)}
for c in CELLS.values():
    test_cell(*c)
HOLE = (-4.5, -3.3, 2.0)                          # machined hole: centre x, y, radius [mm]
add_circle(*HOLE, 0.08)
ang = np.radians(40)                              # 30 um burr protruding into the hole
add_circle(HOLE[0] + (HOLE[2]) * np.cos(ang), HOLE[1] + HOLE[2] * np.sin(ang), 0.030, BG)
ang = np.radians(200)                             # 20 um nick on the hole edge (missing material)
add_circle(HOLE[0] + (HOLE[2] + 0.01) * np.cos(ang), HOLE[1] + (HOLE[2] + 0.01) * np.sin(ang), 0.020, 0.08)
add_rect(4.6, 3.6, 1.2, 4.5, 0.08)                # slot
add_rect(4.6, 3.6 - 2.25 + 0.006, 0.3, 0.012, BG)  # 12 um burr at the slot end


def coverage(kind, p, X, Y, soft):
    if kind == "circle":
        cx, cy, r = p
        d = np.hypot(X - cx, Y - cy) - r
    else:
        cx, cy, w, h, a = p
        xr = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
        yr = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
        d = np.maximum(np.abs(xr) - w / 2, np.abs(yr) - h / 2)
    return np.clip(0.5 - d / soft, 0, 1)


def render(wl, ideal=False):
    """Scene reflectance sampled on the image grid (real mapping, or ideal |m0| scaling)."""
    img = np.full((u.size, w_.size), BG, np.float32).T.copy()       # rows = y
    soft = S / m0                                                      # object size of one sample
    for kind, p, val, (x0, x1, y0, y1) in features:
        j0, j1_ = np.searchsorted(u, [x0 * m0 - 3 * S, x1 * m0 + 3 * S])
        i0, i1 = np.searchsorted(w_, [y0 * m0 - 3 * S, y1 * m0 + 3 * S])
        if j1_ <= j0 or i1 <= i0:
            continue
        U, V = np.meshgrid(u[j0:j1_], w_[i0:i1])
        X, Y = (U / m0, V / m0) if ideal else image_to_object(U, V, wl)
        c = coverage(kind, p, X, Y, soft)
        img[i0:i1, j0:j1_] = img[i0:i1, j0:j1_] * (1 - c) + val * c
    return img


# ---------------------------------------------------------------- PSFs
t = (np.arange(2 * KH + 1) - KH) * S * 1e3                            # kernel coords [um]
TX, TY = np.meshgrid(t, t)
PSF_R = np.array([0.0, 2.75, 5.5, 7.7, 9.5, 11.0])                    # object radii for PSFs
psf_tab = {}
for wl in WLS:
    for r in PSF_R:
        p = FFTPSF(lens, (0.0, r / T.OBJ_HEIGHTS[-1]), wl, num_rays=64, grid_size=512)
        # Optiland PSF arrays are in image coordinates (cols = x, rows = y); the
        # display removes the 180 deg inversion, so rotate the PSF by 180 deg too
        # (roll keeps the chief ray on the centre sample)
        a = np.roll(np.asarray(p.psf, dtype=float)[::-1, ::-1], 1, axis=(0, 1))
        dx = wl * p._get_working_FNO() / (p.grid_size / (p.num_rays - 1))   # um / PSF sample
        c = a.shape[0] // 2
        psf_tab[wl, r] = (a, dx, c)
    a, dx, c = psf_tab[wl, 0.0]
    yy, xx = np.indices(a.shape)
    print(f"{wl * 1e3:.0f} nm: PSF sample {dx:.3f} um, on-axis centroid offset "
          f"({(np.sum(xx * a) / a.sum() - c) * dx:.2f}, {(np.sum(yy * a) / a.sum() - c) * dx:.2f}) um")
a = psf_tab[WLS[0], 11.0][0]
print(f"Edge PSF (486 nm) asymmetric energy fraction: {np.abs(a - a[::-1, ::-1]).sum() / (2 * a.sum()):.3f}")


def kernel(wl, r_mm, phi):
    """PSF at object radius r_mm, rotated so that its radial axis points at angle phi."""
    i = np.clip(np.searchsorted(PSF_R, r_mm) - 1, 0, len(PSF_R) - 2)
    f = np.clip((r_mm - PSF_R[i]) / (PSF_R[i + 1] - PSF_R[i]), 0, 1)
    # reference PSFs: field along +y of the displayed image (radial direction = +y)
    rot = phi - np.pi / 2
    xs = TX * np.cos(rot) + TY * np.sin(rot)
    ys = -TX * np.sin(rot) + TY * np.cos(rot)
    k = 0
    for j, wt in ((i, 1 - f), (i + 1, f)):
        a, dx, c = psf_tab[wl, PSF_R[j]]
        k = k + wt * map_coordinates(a, [c + ys / dx, c + xs / dx], order=1)
    return k / k.sum()


def airy(wl):
    rr = np.pi * np.hypot(TX, TY) / (wl * T.OBJECT_NA ** -1 * abs(m0) / 2) / 1.0
    rr = np.where(rr == 0, 1e-9, rr)
    k = (2 * j1(rr) / rr) ** 2
    return k / k.sum()


# ---------------------------------------------------------------- spatially variable blur
GX, GY = 11, 9                                                         # PSF nodes
nu, nw = np.linspace(u[0], u[-1], GX), np.linspace(w_[0], w_[-1], GY)


def hat(z, nodes, k):
    d = nodes[1] - nodes[0]
    return np.clip(1 - np.abs(z - nodes[k]) / d, 0, 1)


def blur_variable(img, wl):
    out = np.zeros_like(img)
    for ky in range(GY):
        wy = hat(w_, nw, ky)
        i0, i1 = np.flatnonzero(wy)[[0, -1]] + [0, 1]
        for kx in range(GX):
            wx = hat(u, nu, kx)
            j0, j1_ = np.flatnonzero(wx)[[0, -1]] + [0, 1]
            r_img = np.hypot(nu[kx], nw[ky])
            r_o = np.interp(r_img, rho[wl], r_obj)
            k = kernel(wl, r_o, np.arctan2(nw[ky], nu[kx]))
            a0, a1 = max(i0 - KH, 0), min(i1 + KH, img.shape[0])
            b0, b1 = max(j0 - KH, 0), min(j1_ + KH, img.shape[1])
            conv = fftconvolve(np.pad(img[a0:a1, b0:b1], KH, mode="edge"), k, mode="same")[KH:-KH, KH:-KH]
            conv = conv[i0 - a0:i0 - a0 + i1 - i0, j0 - b0:j0 - b0 + j1_ - j0]
            out[i0:i1, j0:j1_] += conv * np.outer(wy[i0:i1], wx[j0:j1_])
    return out


def sensor(irr, rng):
    """Pixel integration (OS x OS), shot + read noise, 8 bit."""
    px = irr.reshape(NY, OS, NX, OS).mean((1, 3))
    e = rng.poisson(np.clip(px, 0, None) * EXPOSURE * FULL_WELL) + rng.normal(0, READ_NOISE, px.shape)
    return np.clip(np.round(e / FULL_WELL * 255), 0, 255).astype(np.uint8)


U2, V2 = np.meshgrid(u, w_)
ri_map = np.interp(np.hypot(U2, V2) / m0, ri_h, ri).astype(np.float32)
del U2, V2

ours, perfect, ideal = 0, 0, 0
for wl in WLS:
    print(f"simulating {wl * 1e3:.0f} nm ...", flush=True)
    scene = render(wl)
    ours = ours + blur_variable(scene, wl) * ri_map / 3
    scene_i = render(wl, ideal=True)
    perfect = perfect + fftconvolve(np.pad(scene_i, KH, mode="edge"), airy(wl), mode="same")[KH:-KH, KH:-KH] / 3
    ideal = ideal + scene_i / 3

rng = np.random.default_rng(1)
img_ours, img_perfect, img_ideal = (sensor(a, rng) for a in (ours, perfect, ideal))
for name, im in (("lens", img_ours), ("perfect", img_perfect), ("ideal", img_ideal)):
    plt.imsave(f"{prefix}_sim_{name}.png", im[::-1], cmap="gray", vmin=0, vmax=255)

# ---------------------------------------------------------------- figures
ext = [-NX * PIX / 2 / m0, NX * PIX / 2 / m0, -NY * PIX / 2 / m0, NY * PIX / 2 / m0]   # object mm
fig, ax = plt.subplots(figsize=(11, 9))
ax.imshow(img_ours, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=255)
for name, (cx, cy) in CELLS.items():
    ax.add_patch(Rectangle((cx - 0.55, cy - 0.45), 1.1, 0.9, fill=False, ec="tab:orange", lw=1.5))
    ax.text(cx, cy + 0.55, name, color="tab:orange", ha="center", fontsize=10)
ax.add_patch(Rectangle((HOLE[0] - 2.3, HOLE[1] - 2.3), 4.6, 4.6, fill=False, ec="tab:cyan", lw=1.2))
ax.set_xlabel("Object x [mm]"); ax.set_ylabel("Object y [mm]")
ax.set_title(f"Simulated sensor image: 2448 x 2048 px, 3.45 um, white LED (486/588/656 nm)\n"
             f"field of view {ext[1] * 2:.1f} x {ext[3] * 2:.1f} mm at 0.5x")
fig.tight_layout(); fig.savefig(f"{prefix}_sim_full.png", dpi=130)


def crop(im, cx, cy, hw, hh):
    j0, j1_ = ((np.array([cx - hw, cx + hw]) * m0 + NX * PIX / 2) / PIX).astype(int)
    i0, i1 = ((np.array([cy - hh, cy + hh]) * m0 + NY * PIX / 2) / PIX).astype(int)
    return im[i0:i1, j0:j1_], [cx - hw, cx + hw, cy - hh, cy + hh]


cols = (("Ideal (no blur, sensor pixels)", img_ideal), ("Perfect lens, NA 0.028", img_perfect),
        ("Our lens (real rays + PSFs)", img_ours))
fig, axs = plt.subplots(len(CELLS), 3, figsize=(15, 4.3 * len(CELLS)))
for r, (name, (cx, cy)) in enumerate(CELLS.items()):
    for c, (title, im) in enumerate(cols):
        sub, e = crop(im, cx, cy, 0.55, 0.45)
        axs[r, c].imshow(sub, cmap="gray", origin="lower", extent=e, vmin=0, vmax=255, interpolation="nearest")
        axs[r, c].set_title(f"{name} (r = {np.hypot(cx, cy):.1f} mm) - {title}", fontsize=9)
        axs[r, c].tick_params(labelsize=7)
    a = axs[r, 0]
    x = cx - 0.48
    for wb in BAR_W:
        a.text(x + 2.5 * wb * 1e-3, cy + 0.415, f"{wb:g}", color="tab:red",
               fontsize=6, ha="center")
        x += 5 * wb * 1e-3 + 0.06
    for i, d in enumerate(PIT_D):
        a.text(cx - 0.42 + 0.115 * i, cy - 0.10, f"{d:g}", color="tab:red", fontsize=6, ha="center")
    for i, wb in enumerate(SCR_W):
        a.text(cx - 0.36 + 0.18 * i, cy - 0.43, f"{wb:g}", color="tab:red", fontsize=6, ha="center")
fig.suptitle("Test cells: bars (width um, top two rows), pits (diameter um), scratches (width um)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{prefix}_sim_cells.png", dpi=130)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for c, (title, im) in enumerate(cols):
    for k, (dx_, dy_) in enumerate(((np.cos(np.radians(40)), np.sin(np.radians(40))),)):
        bx, by = HOLE[0] + HOLE[2] * dx_, HOLE[1] + HOLE[2] * dy_
        sub, e = crop(im, bx, by, 0.25, 0.2)
        axs[c].imshow(sub, cmap="gray", origin="lower", extent=e, vmin=0, vmax=255, interpolation="nearest")
        axs[c].set_title(f"Hole edge with 30 um burr - {title}", fontsize=9)
fig.tight_layout(); fig.savefig(f"{prefix}_sim_burr.png", dpi=130)


# ---------------------------------------------------------------- contrast numbers
def bar_contrast(im, cx, cy, wb):
    x = cx - 0.48
    for b in BAR_W:
        if b == wb:
            break
        x += 5 * b * 1e-3 + 0.06
    w = wb * 1e-3
    xs = x + w / 2 + np.arange(5) * w                # bar, gap, bar, gap, bar centres
    j = ((xs * m0 + NX * PIX / 2) / PIX).astype(int)
    i = int((cy + 0.30) * m0 / PIX + NY / 2)
    prof = im[i - 2:i + 3].astype(float).mean(0)
    lo = prof[j[[0, 2, 4]]].mean()
    hi = prof[j[[1, 3]]].mean()
    return (hi - lo) / (hi + lo)


print("\nBar contrast (vertical bars) [Michelson], object bar width in um")
for wb in BAR_W:
    vals = []
    for name, (cx, cy) in CELLS.items():
        vals += [bar_contrast(img_perfect, cx, cy, wb), bar_contrast(img_ours, cx, cy, wb)]
    print(f"{wb:5.1f} um  " + "  ".join(f"{n}: perfect {vals[2 * k]:.2f} / ours {vals[2 * k + 1]:.2f}"
                                       for k, n in enumerate(CELLS)))
print("saved", f"{prefix}_sim_full.png, _sim_cells.png, _sim_burr.png, _sim_lens/perfect/ideal.png")
