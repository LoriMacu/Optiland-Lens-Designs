"""
0.5x object-space telecentric objective for machine vision (2/3" sensor).

Specification (modelled on commercial 0.5x / 110 mm WD telecentric lenses)
  Magnification -0.50x +-1% | 2/3" sensor: image semi-height 5.5 mm
  object field semi-height 11 mm | working distance 110 mm
  object-side NA 0.028 (~f/9 working) | object-space telecentricity <= 0.10 deg
  distortion <= 0.10 % | image-side chief-ray angle <= 10 deg
  0.486 / 0.588 / 0.656 um equal weights | 3-8 spherical elements, catalog glass
  stop inside the lens | clear diameter <= 40 mm | lens length <= 180 mm
  BFD >= 12 mm | CT >= 2 mm, ET >= 1 mm | P penalises every rule

Goal: best diffraction-limited performance -> minimise the polychromatic RMS
spot on the sensor; judge the result with RMS wavefront error / Strehl and MTF.

First-order layout: front achromat G1 (f ~ 90 mm) with the stop at its rear
focal point (=> chief rays parallel to the axis in object space), rear
achromat G2 (f ~ 50 mm) 5 mm behind the stop forming the -0.5x image.
"""

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from optiland import optimization
from optiland.distribution import HexagonalDistribution
from optiland.optic import Optic
from optiland.optimization.operand import operand_registry
from optiland.optimization.operand.lens import LensOperand
from optiland.optimization.optimizer.scipy import least_squares as _lsq_module
from scipy import optimize as _scipy_optimize

# =============================================================================
# SPECIFICATION
# =============================================================================
MAG_TARGET, MAG_TOL = -0.50, 0.01        # +-1 %
OBJ_HEIGHTS = [0.0, 5.5, 7.7, 9.5, 11.0]  # mm (object semi-heights)
WD = 110.0                               # mm, object to first vertex
OBJECT_NA = 0.028
TELE_MAX = 0.10                          # deg, object-space chief-ray angle
DIST_MAX = 0.10                          # %
CRA_MAX = 10.0                           # deg, image-side chief-ray angle
SEMI_AP_MAX = 20.0                       # mm (clear diameter <= 40 mm)
LENGTH_MAX = 180.0                       # mm, first vertex to image
BFD_MIN = 12.0
CT_MIN, ET_MIN = 2.0, 1.0
N_ELEM = (3, 8)
WAVELENGTHS = [0.486, 0.588, 0.656]
PRIMARY_WL = 0.588
PENALTY_WEIGHT = 1.0
MARGIN = {"mag": 0.003, "tele": 0.03, "dist": 0.02, "cra": 0.5, "semi": 0.3,
          "len": 1.0, "bfd": 0.3, "et": 0.05}
RINGS_OPT, RINGS_FINAL = 6, 16

# =============================================================================
# LENS
# =============================================================================
def _finish(lens):
    lens.set_aperture(aperture_type="objectNA", value=OBJECT_NA)
    lens.fields.set_type("object_height")
    for h in OBJ_HEIGHTS:
        lens.fields.add(y=h)
    for wl in WAVELENGTHS:
        lens.wavelengths.add(value=wl, is_primary=(wl == PRIMARY_WL))
    return lens


def build_start(crown="N-BK7", flint="N-SF5"):
    """Two cemented achromats (G1 f~90, G2 f~50), stop at G1's rear focus."""
    lens = Optic(name="Object-telecentric 0.5x")
    c, f = (crown, "schott"), (flint, "schott")
    lens.surfaces.add(index=0, radius=np.inf, thickness=WD)
    # G1: achromat f ~ 90 mm
    lens.surfaces.add(index=1, radius=46.3, thickness=5.0, material=c)
    lens.surfaces.add(index=2, radius=-46.3, thickness=2.5, material=f)
    lens.surfaces.add(index=3, radius=-202.8, thickness=86.0)
    # stop
    lens.surfaces.add(index=4, radius=np.inf, thickness=5.0, is_stop=True)
    # G2: achromat f ~ 50 mm
    lens.surfaces.add(index=5, radius=25.7, thickness=4.0, material=c)
    lens.surfaces.add(index=6, radius=-25.7, thickness=2.0, material=f)
    lens.surfaces.add(index=7, radius=-112.7, thickness=42.0)
    lens.surfaces.add(index=8)
    return _finish(lens)


def build_flattened(crown="N-BK7", flint="N-SF5", ff_glass="N-SF5"):
    """Optimised 4-element design + a negative field-flattener lens close to
    the image (cancels the Petzval curvature of the two positive groups)."""
    lens = Optic(name="Object-telecentric 0.5x + field flattener")
    c, f, g = (crown, "schott"), (flint, "schott"), (ff_glass, "schott")
    lens.surfaces.add(index=0, radius=np.inf, thickness=WD)
    lens.surfaces.add(index=1, radius=50.2671, thickness=5.4827, material=c)
    lens.surfaces.add(index=2, radius=-40.7576, thickness=2.4756, material=f)
    lens.surfaces.add(index=3, radius=-156.4829, thickness=88.1393)
    lens.surfaces.add(index=4, radius=np.inf, thickness=8.1528, is_stop=True)
    lens.surfaces.add(index=5, radius=24.3711, thickness=4.0351, material=c)
    lens.surfaces.add(index=6, radius=-49.5668, thickness=2.5165, material=f)
    lens.surfaces.add(index=7, radius=-481.2501, thickness=24.0)
    # field flattener: negative lens, concave towards the object
    lens.surfaces.add(index=8, radius=-20.0, thickness=2.5, material=g)
    lens.surfaces.add(index=9, radius=-200.0, thickness=14.0)
    lens.surfaces.add(index=10)
    return _finish(lens)


def build_7el(g1_crown="N-BK7", g1_flint="N-SF5", g2_crown="N-SK16", g2_flint="F2",
              ff_glass="N-SF5"):
    """Front group: weak positive meniscus + achromat; stop at its rear focus;
    rear group: Cooke triplet (f ~ 50 mm, Petzval/astigmatism control);
    negative field flattener near the image."""
    lens = Optic(name="Object-telecentric 0.5x, 7 elements")
    g = lambda name: (name, "schott")
    lens.surfaces.add(index=0, radius=np.inf, thickness=WD)
    # front group
    lens.surfaces.add(index=1, radius=120.0, thickness=4.0, material=g(g1_crown))
    lens.surfaces.add(index=2, radius=400.0, thickness=1.0)
    lens.surfaces.add(index=3, radius=50.27, thickness=5.5, material=g(g1_crown))
    lens.surfaces.add(index=4, radius=-40.76, thickness=2.5, material=g(g1_flint))
    lens.surfaces.add(index=5, radius=-156.5, thickness=72.0)
    # stop
    lens.surfaces.add(index=6, radius=np.inf, thickness=5.0, is_stop=True)
    # rear group: Cooke triplet (Optiland sample radii, f = 50 mm)
    lens.surfaces.add(index=7, radius=22.01, thickness=3.26, material=g(g2_crown))
    lens.surfaces.add(index=8, radius=-435.76, thickness=6.0)
    lens.surfaces.add(index=9, radius=-22.21, thickness=2.0, material=g(g2_flint))
    lens.surfaces.add(index=10, radius=20.29, thickness=4.75)
    lens.surfaces.add(index=11, radius=79.68, thickness=2.95, material=g(g2_crown))
    lens.surfaces.add(index=12, radius=-18.40, thickness=25.0)
    # field flattener
    lens.surfaces.add(index=13, radius=-30.0, thickness=2.5, material=g(ff_glass))
    lens.surfaces.add(index=14, radius=-200.0, thickness=13.0)
    lens.surfaces.add(index=15)
    return _finish(lens)


def elements(optic):
    out = []
    for i in range(1, optic.surfaces.num_surfaces - 1):
        if float(np.squeeze(optic.surfaces[i].material_post.n(PRIMARY_WL))) > 1.01:
            out.append((i, i + 1))
    return out


# =============================================================================
# METRICS (cached per lens state: the optimiser asks for many operands)
# =============================================================================
_cache = {}


def _key(optic):
    n = optic.surfaces.num_surfaces
    z = np.asarray(optic.surfaces.positions).ravel()[1:]
    return tuple(float(np.squeeze(optic.surfaces[i].geometry.radius)) for i in range(1, n)) + \
        tuple(np.round(z, 12)) + (id(optic),)


def thicknesses(optic):
    z = np.asarray(optic.surfaces.positions).ravel()
    return np.diff(z[1:])              # thickness after surface 1, 2, ...


def spot_sizes(optic, rings=RINGS_OPT):
    """Polychromatic RMS spot radius (mm) per field, about the common centroid."""
    key = ("spot", rings) + _key(optic)
    if _cache.get("spot_key") == key:
        return _cache["spot"]
    d = HexagonalDistribution()
    d.generate_points(rings)
    px, py = np.asarray(d.x), np.asarray(d.y)
    nf, npup = len(OBJ_HEIGHTS), len(px)
    Hy = np.repeat(np.array(OBJ_HEIGHTS) / OBJ_HEIGHTS[-1], npup)
    Px, Py = np.tile(px, nf), np.tile(py, nf)
    xs, ys = [], []
    for wl in WAVELENGTHS:
        r = optic.trace_generic(np.zeros_like(Hy), Hy, Px, Py, wl)
        xs.append(np.asarray(r.x).reshape(nf, npup))
        ys.append(np.asarray(r.y).reshape(nf, npup))
    x, y = np.concatenate(xs, 1), np.concatenate(ys, 1)
    rms = np.sqrt(np.mean((x - x.mean(1, keepdims=True)) ** 2 + (y - y.mean(1, keepdims=True)) ** 2, 1))
    rms = np.where(np.isfinite(rms), rms, 10.0)
    _cache["spot_key"], _cache["spot"] = key, rms
    return rms


def chief_ray_data(optic, near_axis=0.05):
    """Real chief rays (primary wavelength): object-space angle, image-side
    angle, image heights. Index 0 is a near-axis ray used for the reference
    magnification."""
    h = np.array([near_axis] + OBJ_HEIGHTS[1:])
    Hy = h / OBJ_HEIGHTS[-1]
    z0 = np.zeros_like(Hy)
    optic.trace_generic(z0, Hy, z0, z0, PRIMARY_WL)
    M = np.asarray(optic.surfaces.M)      # y direction cosines after each surface
    N = np.asarray(optic.surfaces.N)
    y = np.asarray(optic.surfaces.y)
    # atan(M/N), not atan2: near telecentricity the entrance pupil is at +-"infinity"
    # and Optiland may trace the chief ray with reversed direction cosines
    # (same line, opposite sense); M/N is the same in both cases.
    obj_angle = np.degrees(np.arctan(M[0] / N[0]))      # leaving the object
    img_angle = np.degrees(np.arctan(M[-2] / N[-2]))    # arriving at the image
    y_img = y[-1]
    mag0 = y_img[0] / h[0]
    dist = 100 * (y_img[1:] - mag0 * h[1:]) / (mag0 * h[1:])
    return {"mag": float(mag0), "tele": obj_angle[1:], "cra": img_angle[1:],
            "dist": dist, "y_img": y_img[1:]}


def system_values(optic):
    key = ("sys",) + _key(optic)
    if _cache.get("sys_key") == key:
        return _cache["sys"]
    optic.updater.update_paraxial()      # refresh semi-apertures (|marginal|+|chief|)
    t = thicknesses(optic)
    elems = elements(optic)
    stop = optic.surfaces.stop_index
    cr = chief_ray_data(optic)
    n = optic.surfaces.num_surfaces
    semi = [float(np.squeeze(optic.surfaces[i].semi_aperture)) for i in range(1, n - 1)]
    v = {
        "mag": cr["mag"],
        "tele": [0.0] + list(cr["tele"]),
        "cra": [0.0] + list(cr["cra"]),
        "dist": [0.0] + list(cr["dist"]),
        "wd": WD,                        # object thickness is fixed (not a variable)
        "length": float(t.sum()),
        "bfd": float(t[-1]),
        "semi": semi,
        "ct": [float(t[a - 1]) for a, _ in elems],
        "et": [float(np.squeeze(LensOperand.edge_thickness(optic, a))) for a, _ in elems],
        "n_elements": len(elems),
        "stop_inside": elems[0][0] < stop < elems[-1][1],
        "working_fno": 1 / (2 * OBJECT_NA / abs(cr["mag"])),
    }
    _cache["sys_key"], _cache["sys"] = key, v
    return v


def violations(optic, margin=True):
    m = MARGIN if margin else {k: 0.0 for k in MARGIN}
    v = system_values(optic)
    return {
        "Magnification": max(0.0, abs(v["mag"] / MAG_TARGET - 1) - (MAG_TOL - m["mag"])) / MAG_TOL,
        "Telecentricity": sum(max(0.0, abs(a) - (TELE_MAX - m["tele"])) for a in v["tele"]) / TELE_MAX,
        "Distortion": sum(max(0.0, abs(d) - (DIST_MAX - m["dist"])) for d in v["dist"]) / DIST_MAX,
        "Image CRA": sum(max(0.0, abs(a) - (CRA_MAX - m["cra"])) for a in v["cra"]) / CRA_MAX,
        "Clear diameter": sum(max(0.0, s - (SEMI_AP_MAX - m["semi"])) for s in v["semi"]) / SEMI_AP_MAX,
        "Lens length": max(0.0, v["length"] - (LENGTH_MAX - m["len"])) / LENGTH_MAX,
        "BFD": max(0.0, BFD_MIN + m["bfd"] - v["bfd"]) / BFD_MIN,
        "Center thickness": sum(max(0.0, CT_MIN - t) for t in v["ct"]) / CT_MIN,
        "Edge thickness": sum(max(0.0, ET_MIN + m["et"] - t) for t in v["et"]) / ET_MIN,
        "Elements (3-8)": float(not N_ELEM[0] <= v["n_elements"] <= N_ELEM[1]),
        "Stop inside": float(not v["stop_inside"]),
    }


def penalty(optic, margin=True):
    return sum((PENALTY_WEIGHT * x) ** 2 for x in violations(optic, margin).values())


def merit(optic, rings=RINGS_OPT):
    return float(np.sum(spot_sizes(optic, rings) ** 2)) + penalty(optic)


# Operands for Optiland's optimiser
operand_registry.register("tc_spot", lambda optic, k: float(spot_sizes(optic)[k]), overwrite=True)
operand_registry.register("tc_constraint", lambda optic, name: violations(optic)[name], overwrite=True)
operand_registry.register("tc_tele", lambda optic, k: float(system_values(optic)["tele"][k]), overwrite=True)
operand_registry.register("tc_dist", lambda optic, k: float(system_values(optic)["dist"][k]), overwrite=True)
operand_registry.register("tc_mag", lambda optic: float(system_values(optic)["mag"]), overwrite=True)


# =============================================================================
# OPTIMISER
# =============================================================================
def _least_squares_compat(*args, callback=None, **kwargs):
    # SciPy < 1.16 has no `callback`; x_scale="jac" balances radii vs thicknesses
    return _scipy_optimize.least_squares(*args, x_scale="jac", **kwargs)


_lsq_module.optimize = SimpleNamespace(least_squares=_least_squares_compat)


def build_problem(lens):
    p = optimization.OptimizationProblem()
    nf = len(OBJ_HEIGHTS)
    for k in range(nf):
        p.add_operand("tc_spot", target=0.0, weight=1.0, input_data={"optic": lens, "k": k})
    # smooth targets (the one-sided P terms have kinks that stall the solver)
    for k in range(1, nf):
        p.add_operand("tc_tele", target=0.0, weight=0.05, input_data={"optic": lens, "k": k})
        p.add_operand("tc_dist", target=0.0, weight=0.02, input_data={"optic": lens, "k": k})
    p.add_operand("tc_mag", target=MAG_TARGET, weight=1.0, input_data={"optic": lens})
    for name in violations(lens):
        p.add_operand("tc_constraint", target=0.0, weight=PENALTY_WEIGHT,
                      input_data={"optic": lens, "name": name})

    n = lens.surfaces.num_surfaces
    stop = lens.surfaces.stop_index
    for i in range(1, n - 1):
        if i != stop:
            p.add_variable(lens, "radius", surface_number=i)
    glass = {a for a, _ in elements(lens)}
    for i in range(1, n - 2):
        if i in glass:
            p.add_variable(lens, "thickness", surface_number=i, min_val=CT_MIN, max_val=15.0)
        else:
            p.add_variable(lens, "thickness", surface_number=i, min_val=0.5, max_val=120.0)
    p.add_variable(lens, "thickness", surface_number=n - 2, min_val=BFD_MIN, max_val=100.0)
    return p


def optimize(lens, max_blocks=12, nfev_per_block=25, rel_tol=3e-3, verbose=True):
    problem = build_problem(lens)
    last = merit(lens)
    for _ in range(max_blocks):
        optimization.LeastSquares(problem).optimize(maxiter=nfev_per_block, tol=1e-12,
                                                    method_choice="trf")
        now = merit(lens)
        if verbose:
            print(f"    block: merit = {now:.4e}  mean RMS = {np.mean(spot_sizes(lens)) * 1e3:.2f} um  "
                  f"P = {penalty(lens, margin=False):.2e}", flush=True)
        if last - now < rel_tol * last:
            break
        last = now
    return lens


# =============================================================================
# REPORT
# =============================================================================
def rms_wavefront(optic):
    """RMS wavefront error (waves) per field and wavelength, centroid reference."""
    from optiland.wavefront import OPD
    out = np.zeros((len(OBJ_HEIGHTS), len(WAVELENGTHS)))
    for i, h in enumerate(OBJ_HEIGHTS):
        for j, wl in enumerate(WAVELENGTHS):
            opd = OPD(optic, (0.0, h / OBJ_HEIGHTS[-1]), wl, num_rays=16, strategy="centroid")
            out[i, j] = float(np.squeeze(opd.rms()))
    return out


def report(lens):
    v = system_values(lens)
    spots = spot_sizes(lens, RINGS_FINAL) * 1e3
    wfe = rms_wavefront(lens)
    strehl = np.exp(-(2 * np.pi * wfe) ** 2)
    airy = 1.22 * PRIMARY_WL * v["working_fno"]

    print(f"\n=== {lens.name} ===")
    rows = []
    for i in range(1, lens.surfaces.num_surfaces):
        s = lens.surfaces[i]
        last = i == lens.surfaces.num_surfaces - 1
        mat = s.material_post
        rows.append({
            "Surface": "IMA" if last else i,
            "Radius [mm]": float(np.squeeze(s.geometry.radius)),
            "Thickness [mm]": np.nan if last else thicknesses(lens)[i - 1],
            "Glass": getattr(mat, "name", "") if float(np.squeeze(mat.n(PRIMARY_WL))) > 1.01 else "",
            "Semi-ap [mm]": float(np.squeeze(s.semi_aperture)),
            "Stop": "STOP" if s.is_stop else "",
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.4f}".format))

    print(f"\nWorking F/# = {v['working_fno']:.2f}   Airy radius (588 nm) = {airy:.2f} um")
    perf = pd.DataFrame({
        "Object h [mm]": OBJ_HEIGHTS,
        "Poly RMS spot [um]": spots,
        "RMS WFE 486 [waves]": wfe[:, 0], "RMS WFE 588": wfe[:, 1], "RMS WFE 656": wfe[:, 2],
        "Strehl (min)": strehl.min(1),
        "Telecentricity [deg]": v["tele"],
        "Distortion [%]": v["dist"],
        "Image CRA [deg]": v["cra"],
    })
    print(perf.to_string(index=False, float_format="{:.4f}".format))
    print(f"Mean RMS spot = {spots.mean():.3f} um   worst RMS WFE = {wfe.max():.4f} waves "
          f"(diffraction-limited if <= 0.0714)   min Strehl = {strehl.min():.3f}")

    checks = [
        ("Magnification", f"{v['mag']:.4f}", "-0.495 ... -0.505", abs(v["mag"] / MAG_TARGET - 1) <= MAG_TOL),
        ("Working distance [mm]", f"{v['wd']:.1f}", "110 +-1", True),
        ("Object NA", f"{OBJECT_NA}", "0.028", True),
        ("Max telecentricity [deg]", f"{max(abs(a) for a in v['tele']):.4f}", "<= 0.10",
         max(abs(a) for a in v["tele"]) <= TELE_MAX),
        ("Max |distortion| [%]", f"{max(abs(d) for d in v['dist']):.4f}", "<= 0.10",
         max(abs(d) for d in v["dist"]) <= DIST_MAX),
        ("Max image CRA [deg]", f"{max(abs(a) for a in v['cra']):.2f}", "<= 10", max(abs(a) for a in v["cra"]) <= CRA_MAX),
        ("Max clear diameter [mm]", f"{2 * max(v['semi']):.2f}", "<= 40", max(v["semi"]) <= SEMI_AP_MAX),
        ("Lens length [mm]", f"{v['length']:.2f}", "<= 180", v["length"] <= LENGTH_MAX),
        ("BFD [mm]", f"{v['bfd']:.2f}", ">= 12", v["bfd"] >= BFD_MIN),
        ("Elements", f"{v['n_elements']}", "3 - 8", N_ELEM[0] <= v["n_elements"] <= N_ELEM[1]),
        ("Stop inside", f"{v['stop_inside']}", "True", v["stop_inside"]),
        ("Min CT [mm]", f"{min(v['ct']):.3f}", ">= 2", min(v["ct"]) >= CT_MIN),
        ("Min ET [mm]", f"{min(v['et']):.3f}", ">= 1", min(v["et"]) >= ET_MIN),
    ]
    print(pd.DataFrame(checks, columns=["Rule", "Value", "Limit", "OK"]).to_string(index=False))
    P = penalty(lens, margin=False)
    print(f"Penalty P (strict) = {P:.3e}   Score = sum(RMS^2) + P = "
          f"{np.sum((spots * 1e-3) ** 2) + P:.4e} mm^2")
    return spots, wfe, P
