#!/usr/bin/env python3
r"""
focusing.py
===========
Quantitative dike-focusing calculation for TuLIP section 2.5.

Physical idea
-------------
An inflating magma reservoir perturbs the crustal stress field. Ascending dikes
follow sigma1 (max-compression) trajectories (Anderson 1951; Nakamura 1977). Over
a region around the reservoir the perturbation dominates the ambient differential
stress and reorients sigma1 so that dikes are steered toward the reservoir /
summit ("stress umbrella"; Muller et al. 2001; Karlstrom et al. 2009; Pinel &
Jaupart 2004). We quantify the horizontal radius R_f of the captured region and
the focusing area A_f = pi R_f^2, replacing the manuscript's ad-hoc "sills within
3x semi-major axis merge" rule with a stress-derived length scale.

Two independent estimates of R_f, required to agree:
  (1) CLOSED FORM.  Reservoir differential stress ~ dP a^3 / R^3 balances the
      ambient differential stress Dsig_bg at a cross-over radius
          R_c = a (C * dP / Dsig_bg)^(1/3),
      C an O(1) geometric constant calibrated once against (2). This gives the
      scaling law R_f ∝ dP^(1/3), hence, with viscoelastic dP(t)=dP0 e^{-t/tau},
          R_f(t) = R_f0 exp(-t/(3 tau)),   A_f(t) = A_f0 exp(-2 t/(3 tau)).
  (2) TRAJECTORY CAPTURE.  Integrate sigma1 trajectories upward from a seed depth
      z0 below the reservoir; a dike seeded at horizontal offset r0 is "captured"
      if its trajectory reaches the summit window |r| < w at the surface. R_f is
      the largest r0 that is captured.

Implementation notes
--------------------
* RK4 integration of the sigma1 unit-vector field with eigenvector sign
  continuity enforced along each trajectory (no branch flips).
* Everything vectorized; a full R_f(dP) sweep is milliseconds.
"""
from __future__ import annotations
import math
import numpy as np
from stress_field import (reservoir_stress, background_stress, principal_2d,
                          differential_stress, RHO_CRUST, G_GRAV)


# ======================================================================
# 1.  sigma1 trajectory integration (dike path)   -- pure-float, fast
# ======================================================================
def _sigma1_unit(r, z, dP, a, d, K0, sigma_tect, free_surface, rho, g, prev=None):
    r"""
    sigma1 (max-compression) unit vector (er,ez) of the TOTAL field at a single
    point, computed with scalar arithmetic (no numpy overhead). Sign made
    continuous with `prev` (previous step direction) to avoid eigenvector flips.
    """
    ur = r; uz = z + d
    R2 = ur*ur + uz*uz
    if R2 <= a*a:                      # inside reservoir: undefined, return up
        return (0.0, 1.0)
    R = math.sqrt(R2)
    R3 = R2*R
    c = dP*a**3/(2.0*R3)
    nr = ur/R; nz = uz/R
    Srr = c*(3*nr*nr - 1.0)
    Szz = c*(3*nz*nz - 1.0)
    Srz = c*(3*nr*nz)
    if free_surface:
        uz2 = z - d
        R2b = ur*ur + uz2*uz2
        Rb = math.sqrt(R2b); R3b = R2b*Rb
        cb = dP*a**3/(2.0*R3b)
        nrb = ur/Rb; nzb = uz2/Rb
        Srr -= cb*(3*nrb*nrb - 1.0)
        Szz -= cb*(3*nzb*nzb - 1.0)
        Srz -= cb*(3*nrb*nzb)
    depth = -z if z < 0 else 0.0
    Sv = rho*g*depth
    Srr += K0*Sv + sigma_tect
    Szz += Sv
    mean = 0.5*(Srr + Szz)
    diff = 0.5*(Srr - Szz)
    rad = math.sqrt(diff*diff + Srz*Srz)
    s1 = mean + rad
    vr = Srz; vz = s1 - Srr
    nrm = math.hypot(vr, vz)
    if nrm < 1e-30*max(abs(s1), 1.0):
        er, ez = 0.0, 1.0
    else:
        er, ez = vr/nrm, vz/nrm
    if prev is None:
        if ez < 0:
            er, ez = -er, -ez
    else:
        if er*prev[0] + ez*prev[1] < 0:
            er, ez = -er, -ez
    return er, ez


def trace_dike(r0, z0, dP, a, d, K0=1.0, sigma_tect=0.0, free_surface=False,
               rho=RHO_CRUST, g=G_GRAV, ds=None, z_top=0.0, max_steps=None,
               L_max=None, r_domain=None, r_reservoir_capture=True):
    r"""
    Integrate one sigma1 trajectory (dike path) from (r0, z0) upward until it
    reaches the surface z_top, is captured by the reservoir, leaves the domain,
    or exceeds a maximum path length L_max (declared escaped).

    Returns dict with 'r','z' arrays, 'r_surface' (horizontal position at z_top;
    np.nan if captured into reservoir), 'captured_reservoir' bool.
    """
    if ds is None:
        ds = a / 10.0
    if L_max is None:
        L_max = 6.0*(d + abs(z0))          # bounded path length
    if r_domain is None:
        r_domain = 10.0*(d + a)
    if max_steps is None:
        max_steps = int(L_max/ds) + 10
    r = float(r0); z = float(z0)
    rs = [r]; zs = [z]
    prev = None
    for _ in range(max_steps):
        # RK4 on unit sigma1 field
        k1 = _sigma1_unit(r, z, dP, a, d, K0, sigma_tect, free_surface, rho, g, prev)
        k2 = _sigma1_unit(r+0.5*ds*k1[0], z+0.5*ds*k1[1], dP, a, d, K0, sigma_tect, free_surface, rho, g, k1)
        k3 = _sigma1_unit(r+0.5*ds*k2[0], z+0.5*ds*k2[1], dP, a, d, K0, sigma_tect, free_surface, rho, g, k2)
        k4 = _sigma1_unit(r+ds*k3[0], z+ds*k3[1], dP, a, d, K0, sigma_tect, free_surface, rho, g, k3)
        dr = (ds/6.0)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        dz = (ds/6.0)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        prev = (dr, dz)
        r_new = r + dr; z_new = z + dz
        if r_new < 0:            # reflect at axis of symmetry
            r_new = -r_new
        # capture by reservoir?
        dist = math.hypot(r_new, z_new + d)
        if r_reservoir_capture and dist <= a:
            rs.append(r_new); zs.append(z_new)
            return dict(r=np.array(rs), z=np.array(zs), r_surface=np.nan,
                        captured_reservoir=True)
        # reached surface?
        if z_new >= z_top:
            frac = (z_top - z)/(z_new - z) if z_new != z else 0.0
            r_surf = r + frac*dr
            rs.append(r_surf); zs.append(z_top)
            return dict(r=np.array(rs), z=np.array(zs), r_surface=abs(r_surf),
                        captured_reservoir=False)
        # left domain or exceeded path length -> escaped
        if r_new > r_domain or (len(rs)*ds) > L_max:
            rs.append(r_new); zs.append(z_new)
            return dict(r=np.array(rs), z=np.array(zs), r_surface=abs(r_new),
                        captured_reservoir=False)
        r, z = r_new, z_new
        rs.append(r); zs.append(z)
    return dict(r=np.array(rs), z=np.array(zs), r_surface=abs(r),
                captured_reservoir=False)


# ======================================================================
# 2.  Focusing radius by trajectory capture
# ======================================================================
def focusing_radius_trajectory(dP, a, d, z0, K0=1.0, sigma_tect=0.0,
                               free_surface=False, w=None, n_seed=60,
                               r0_max=None, rho=RHO_CRUST, g=G_GRAV, ds=None):
    r"""
    Largest seed offset r0 (at depth z0) whose dike is 'focused': captured by the
    reservoir OR landing within the summit window |r_surface| < w.

    Returns (R_f, seed_offsets, focused_mask, r_surface_array).
    """
    if w is None:
        w = a                      # summit capture half-width = reservoir radius
    if r0_max is None:
        r0_max = 8.0 * a
    seeds = np.linspace(0.0, r0_max, n_seed)
    focused = np.zeros(n_seed, dtype=bool)
    r_surf = np.full(n_seed, np.nan)
    for i, r0 in enumerate(seeds):
        out = trace_dike(r0, z0, dP, a, d, K0, sigma_tect, free_surface,
                         rho, g, ds=ds)
        if out['captured_reservoir']:
            focused[i] = True
        else:
            r_surf[i] = out['r_surface']
            focused[i] = out['r_surface'] < w
    # R_f = largest seed that is focused with all smaller seeds also focused
    # (contiguous focused core around the axis)
    R_f = 0.0
    for i in range(n_seed):
        if focused[i]:
            R_f = seeds[i]
        else:
            break
    return R_f, seeds, focused, r_surf


# ======================================================================
# 3.  Closed-form cross-over radius (DERIVED, not fitted)
# ======================================================================
def reservoir_differential_stress(R, dP, a):
    r"""
    In-plane differential stress (sigma1 - sigma3) of the reservoir field at
    radial distance R. In the meridional plane the source-to-point unit vector
    obeys nr^2+nz^2=1, so the two in-plane principal stresses are
        sigma_RR = +dP a^3/R^3,  sigma_TT_inplane = -dP a^3/(2 R^3),
    giving  Dsig_res(R) = (3/2) dP a^3 / R^3,  independent of angle.
    """
    return 1.5 * dP * a**3 / np.asarray(R, float)**3


def focusing_radius_closed_form(dP, a, Dsig_bg, C=1.5):
    r"""
    Focusing (capture) radius from the stress cross-over: the reservoir
    reorients the principal stresses out to the radius where its differential
    stress equals the ambient differential stress,
        (3/2) dP a^3 / R_f^3 = Dsig_bg
      =>  R_f = a (C dP / Dsig_bg)^(1/3),  with C = 3/2 DERIVED analytically.
    A_f = pi R_f^2. Returns R_f (m). C is exposed only for sensitivity tests;
    the physical value is 3/2.
    """
    return a * (C * dP / Dsig_bg)**(1.0/3.0)


def background_differential_stress(depth, K0=1.0, sigma_tect=0.0,
                                   rho=RHO_CRUST, g=G_GRAV):
    r"""Ambient |sigma_v - sigma_h| = |1-K0| rho g depth + |sigma_tect| (Pa)."""
    Sv = rho*g*np.asarray(depth, float)
    return np.abs(Sv*(1.0-K0)) + abs(sigma_tect)


def background_differential_stress_shellavg(d, K0=1.0, sigma_tect=0.0,
                                            rho=RHO_CRUST, g=G_GRAV, R_c=None):
    r"""
    SHELL-AVERAGED ambient differential stress -- the physically correct single
    value of Dsig_bg to insert into the cross-over law R_c = a(C dP/Dsig_bg)^(1/3).

    Derivation. The closed-form focusing radius is defined by the CROSS-OVER
    SURFACE where the reservoir differential stress equals the ambient one,
    Dsig_res(R)=Dsig_bg. Because Dsig_res(R)=(3/2)dP a^3/R^3 is a function of the
    source-centre distance R alone (angle-independent, §E4), that surface is a
    SPHERE of radius R_c centred on the source at depth d. The correct scalar to
    balance against it is therefore Dsig_bg averaged over that sphere, not at any
    single point. For a lithostatic (depth-linear) background,
        Dsig_bg(z) = |1-K0| rho g |z|  (+ |sigma_tect|),
    parametrize the sphere by z = -d + R_c cos(theta). Provided d > R_c the whole
    sphere lies below the free surface (z<0), so |z| = d - R_c cos(theta) and
        <Dsig_bg>_shell = |1-K0| rho g <|z|>_sphere = |1-K0| rho g d,
    since <cos theta>=0 over the sphere. The angular integral of the linear part
    cancels EXACTLY: the shell-average equals the value at the source depth d.
    Hence evaluating Dsig_bg at z=d is not an ad-hoc single-depth choice -- it IS
    the sphere-average, exact to the extent the background is depth-linear and
    d>R_c. (If d<R_c the sphere clips the surface and a correction ~O((R_c-d)^2)
    appears; flagged below.) A uniform tectonic term |sigma_tect| is unaffected by
    averaging and passes through unchanged.

    R_c (optional) is only used to warn when d < R_c (shell clips the surface).
    """
    val = abs(1.0-K0)*rho*g*float(d) + abs(sigma_tect)
    if R_c is not None and R_c > d:
        import warnings
        warnings.warn(
            f"shell radius R_c={R_c:.0f} m exceeds source depth d={d:.0f} m: the "
            "cross-over sphere clips the free surface, so <Dsig_bg>_shell != "
            "Dsig_bg(d); correction O((R_c-d)^2) neglected.", RuntimeWarning,
            stacklevel=2)
    return val


def background_differential_stress_pathavg(traj, K0=1.0, sigma_tect=0.0,
                                           rho=RHO_CRUST, g=G_GRAV):
    r"""
    ARC-LENGTH average of Dsig_bg along a supplied dike trajectory
    (dict with 'r','z' arrays, as returned by trace_dike).

    DIAGNOSTIC ONLY -- this is NOT the correct average for the cross-over law.
    A dike seeded well below the source traverses a long deep segment where
    Dsig_bg is large but the reservoir field is negligible; arc-length averaging
    therefore over-weights that segment and OVER-estimates the effective ambient
    differential stress (hence UNDER-estimates R_c). The correct average for the
    reorientation cross-over is the shell average
    (background_differential_stress_shellavg), which for a depth-linear background
    reduces to the value at the source depth. Provided here to quantify that bias.
    """
    z = np.asarray(traj['z'], float)
    r = np.asarray(traj['r'], float)
    ds = np.hypot(np.diff(r), np.diff(z))
    zmid = 0.5*(z[:-1] + z[1:])
    Dbg = np.abs(1.0-K0)*rho*g*np.abs(zmid) + abs(sigma_tect)
    if ds.sum() <= 0:
        return float(Dbg[0]) if Dbg.size else np.nan
    return float(np.sum(Dbg*ds)/np.sum(ds))


def calibrate_C(dP, a, d, z0, Dsig_bg, **kw):
    """Calibrate C so closed-form matches the trajectory R_f at one reference state."""
    R_f, *_ = focusing_radius_trajectory(dP, a, d, z0, **kw)
    if R_f <= 0:
        return np.nan, R_f
    C = (R_f/a)**3 * Dsig_bg / dP
    return C, R_f


# ======================================================================
# 4.  Viscoelastic time evolution of the focusing area
# ======================================================================
def focusing_area_vs_time(t, dP0, a, Dsig_bg, relax_modulus, G0, C=1.0):
    r"""
    A_f(t) = pi R_f(t)^2 with R_f(t) = a (C dP(t)/Dsig_bg)^(1/3),
    dP(t) = dP0 * relax_modulus(t)/G0  (fixed-volume correspondence: dP ∝ G(t)).

    relax_modulus : callable t -> G(t) (Pa), any rheology (Maxwell, Burgers, ...).
    Returns (dP_t, R_f_t, A_f_t).
    """
    Gt = np.asarray(relax_modulus(t), float)
    dP_t = dP0 * Gt / G0
    R_f_t = a * (C * dP_t / Dsig_bg)**(1.0/3.0)
    A_f_t = np.pi * R_f_t**2
    return dP_t, R_f_t, A_f_t
