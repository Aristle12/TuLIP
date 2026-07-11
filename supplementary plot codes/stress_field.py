#!/usr/bin/env python3
r"""
stress_field.py
===============
Axisymmetric stress field of a pressurized magma reservoir plus a lithostatic /
tectonic background, and the principal-stress machinery used to quantify dike
focusing (TuLIP manuscript, revised section 2.5).

SIGN CONVENTION
---------------
COMPRESSION-POSITIVE (standard in volcanology / structural geology).
Principal stresses ordered  sigma1 >= sigma2 >= sigma3.
A dike is an opening-mode crack: it opens against sigma3 (least compressive)
and its plane contains sigma1 and sigma2. In the 2-D meridional (r,z) plane the
dike path is everywhere tangent to the sigma1 (most compressive) direction, i.e.
it follows a "sigma1 trajectory" (Anderson 1951; Nakamura 1977; Muller et al.
2001).

GEOMETRY
--------
Axisymmetric cylindrical (r >= 0 horizontal, z positive UP). Reservoir centre at
(r=0, z=-d) where d>0 is the source depth below the free surface z=0. The
elastic solution used here is the FULL-SPACE center-of-dilatation / finite-sphere
exterior field; the free-surface correction (half-space, Mogi 1958) is provided
as an optional first-order image term and is OFF by default (documented in
THEORY.md as an approximation with its validity range d/a >~ 2).

RESERVOIR FIELD (compression-positive, verified in derive_stress_field.py)
--------------------------------------------------------------------------
With source point S, field point P, vector u = P - S, R = |u|, unit n = u/R,
overpressure dP (above lithostatic), radius a:

    sigma_res_ij = (dP a^3 / (2 R^3)) * (3 n_i n_j - delta_ij)          [Pa]

giving radial compression  sigma_RR = +dP a^3/R^3  and hoop tension
sigma_TT = -dP a^3/(2 R^3). (Signs flipped relative to the tension-positive
tensor in the derivation because we adopt compression-positive here; the
derivation verified the tension-positive form, this is its negative.)

All functions are vectorized over numpy arrays of field points.
"""
from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------
# Physical-constant defaults (documented; override via arguments)
# ----------------------------------------------------------------------
RHO_CRUST = 2700.0     # kg m^-3   host-rock density
G_GRAV    = 9.81       # m s^-2
G_SHEAR   = 20e9       # Pa        elastic shear modulus (unrelaxed G0)
NU        = 0.25       # -         Poisson ratio


# ======================================================================
# 1.  Reservoir stress field (axisymmetric, full space)
# ======================================================================
def reservoir_stress(r, z, dP, a, d, free_surface=False, nu=NU):
    r"""
    Stress tensor components of a pressurized spherical reservoir.

    Parameters
    ----------
    r, z : array_like
        Field-point cylindrical coords (m). z positive up, surface at z=0.
    dP : float
        Reservoir overpressure above lithostatic (Pa, >0 = inflation).
    a : float
        Reservoir radius (m).
    d : float
        Source depth (m, >0). Centre at (0, -d).
    free_surface : bool
        If True, add the first-order Mogi free-surface image term (approximate,
        valid d/a >~ 2). Default False (pure full space, exact).
    nu : float
        Poisson ratio (only used by the free-surface image term).

    Returns
    -------
    Srr, Szz, Srz : ndarray
        In-plane stress components (Pa), compression-positive. Srz is the
        shear on the (r,z) faces. (Hoop sigma_tt returned by reservoir_hoop.)
    """
    r = np.asarray(r, float); z = np.asarray(z, float)
    # vector from source (0,-d) to field point (r,z)
    ur = r
    uz = z + d
    R2 = ur*ur + uz*uz
    R = np.sqrt(R2)
    R = np.where(R == 0, np.nan, R)
    nr = ur / R
    nz = uz / R
    pref = dP * a**3 / R**3                      # = dP a^3/R^3
    # compression-positive: sigma_ij = pref*(3 n_i n_j - delta_ij)... but that
    # form has sigma_RR = pref*(3-1)?? No: project. Build Cartesian-in-plane.
    # sigma_ij = (dP a^3 /(2 R^3)) (3 n_i n_j - d_ij)   [derived, tension-pos negated]
    c = dP * a**3 / (2.0 * R**3)
    Srr = c * (3*nr*nr - 1.0)
    Szz = c * (3*nz*nz - 1.0)
    Srz = c * (3*nr*nz)
    if free_surface:
        # WARNING (established in benchmark_freesurface.py, finding F2): this single
        # opposite-sign image cancels the NORMAL surface traction sigma_zz(z=0)->0
        # exactly but DOUBLES the SHEAR traction sigma_rz. It is therefore only a
        # partial (normal-traction-only) correction and is NOT a valid traction-free
        # half-space field. A complete solution needs the Mindlin image system
        # (McTigue 1987). Do not use for quantitative half-space work; the focusing
        # calculation uses the exact full-space field (free_surface=False, default).
        import warnings
        warnings.warn(
            "reservoir_stress(free_surface=True) is a normal-traction-only partial "
            "correction (cancels sigma_zz but doubles sigma_rz at z=0); it is NOT a "
            "traction-free half-space field. Use free_surface=False (full space) for "
            "quantitative work. See benchmark_freesurface.py finding F2.",
            RuntimeWarning, stacklevel=2)
        # First-order image source at (0,+d) with opposite sign (partial correction).
        uz2 = z - d
        R2b = ur*ur + uz2*uz2
        Rb = np.sqrt(np.where(R2b == 0, np.nan, R2b))
        nrb = ur/Rb; nzb = uz2/Rb
        cb = dP * a**3 / (2.0 * Rb**3)
        Srr = Srr - cb*(3*nrb*nrb - 1.0)
        Szz = Szz - cb*(3*nzb*nzb - 1.0)
        Srz = Srz - cb*(3*nrb*nzb)
    return Srr, Szz, Srz


def reservoir_hoop(r, z, dP, a, d):
    """Hoop (circumferential) reservoir stress sigma_phiphi (Pa, compression +)."""
    r = np.asarray(r, float); z = np.asarray(z, float)
    R = np.sqrt(r*r + (z+d)**2)
    R = np.where(R == 0, np.nan, R)
    # tension-positive sigma_TT = -dP a^3/(2R^3); compression-positive -> +...
    # tension-pos hoop = +dP a^3/(2R^3) (tension). compression-pos = -dP a^3/(2R^3)
    return -dP * a**3 / (2.0 * R**3)


# ======================================================================
# 2.  Background lithostatic + tectonic stress
# ======================================================================
def background_stress(r, z, K0=1.0, sigma_tect=0.0, rho=RHO_CRUST, g=G_GRAV):
    r"""
    Ambient stress: vertical lithostatic sigma_v = rho g (depth), horizontal
    sigma_h = K0 sigma_v + sigma_tect.  Compression-positive.  depth = -z.

    Returns Srr_bg, Szz_bg, Srz_bg (Srz_bg = 0).
    """
    z = np.asarray(z, float)
    depth = np.maximum(-z, 0.0)
    Sv = rho * g * depth
    Sh = K0 * Sv + sigma_tect
    Srr = Sh * np.ones_like(z)
    Szz = Sv
    Srz = np.zeros_like(z)
    return Srr, Szz, Srz


# ======================================================================
# 3.  Principal stresses & sigma1 orientation in the (r,z) plane
# ======================================================================
def principal_2d(Srr, Szz, Srz):
    r"""
    In-plane principal stresses of the symmetric 2x2 block
        [[Srr, Srz],[Srz, Szz]].
    Returns (s1, s3, theta1) with s1>=s3 (compression-positive) and theta1 the
    angle (radians) of the sigma1 axis measured from the +z (vertical) axis,
    in (-pi/2, pi/2]. A vertical sigma1 gives theta1=0.
    """
    Srr = np.asarray(Srr, float); Szz = np.asarray(Szz, float); Srz = np.asarray(Srz, float)
    mean = 0.5*(Srr + Szz)
    diff = 0.5*(Srr - Szz)
    rad = np.sqrt(diff*diff + Srz*Srz)
    s1 = mean + rad          # most compressive
    s3 = mean - rad          # least compressive
    # eigenvector of s1: direction of max compression.
    # For matrix [[Srr,Srz],[Srz,Szz]] eigenvector for s1:
    #   (Srr - s1) vr + Srz vz = 0  ->  (vr,vz) ~ (Srz, s1 - Srr)  (in (r,z))
    vr = Srz
    vz = s1 - Srr
    # normalize; handle degenerate (isotropic) points
    nrm = np.hypot(vr, vz)
    small = nrm < 1e-30*np.maximum(np.abs(s1), 1.0)
    vr = np.where(small, 0.0, vr/np.where(nrm == 0, 1, nrm))
    vz = np.where(small, 1.0, vz/np.where(nrm == 0, 1, nrm))
    theta1 = np.arctan2(vr, vz)   # angle from vertical
    return s1, s3, theta1


def sigma1_direction(r, z, dP, a, d, K0=1.0, sigma_tect=0.0,
                     free_surface=False, rho=RHO_CRUST, g=G_GRAV):
    """
    Unit vector (er, ez) of the local sigma1 (max-compression) axis at (r,z),
    for the TOTAL field (reservoir + background). Used to integrate dike paths.
    Sign is chosen so ez >= 0 (upward-propagating branch).
    """
    Srr_r, Szz_r, Srz_r = reservoir_stress(r, z, dP, a, d, free_surface=free_surface)
    Srr_b, Szz_b, Srz_b = background_stress(r, z, K0, sigma_tect, rho, g)
    Srr = Srr_r + Srr_b; Szz = Szz_r + Szz_b; Srz = Srz_r + Srz_b
    _, _, th = principal_2d(Srr, Szz, Srz)
    er = np.sin(th); ez = np.cos(th)
    # orient upward
    flip = ez < 0
    er = np.where(flip, -er, er)
    ez = np.where(flip, -ez, ez)
    return er, ez


def differential_stress(r, z, dP, a, d, K0=1.0, sigma_tect=0.0,
                        free_surface=False, rho=RHO_CRUST, g=G_GRAV):
    """Total in-plane differential stress s1 - s3 (Pa)."""
    Srr_r, Szz_r, Srz_r = reservoir_stress(r, z, dP, a, d, free_surface=free_surface)
    Srr_b, Szz_b, Srz_b = background_stress(r, z, K0, sigma_tect, rho, g)
    s1, s3, _ = principal_2d(Srr_r+Srr_b, Szz_r+Szz_b, Srz_r+Srz_b)
    return s1 - s3


# ======================================================================
# 4.  Viscoelastic (Maxwell) time factor  --  fixed-volume relaxation
# ======================================================================
def maxwell_overpressure(t, dP0, eta, G0=G_SHEAR):
    r"""
    dP(t) = dP0 * exp(-t/tauM),  tauM = eta/G0   (verified E1-E3).
    t, eta in SI (s, Pa s). Returns dP(t) in Pa.
    """
    tauM = eta / G0
    return dP0 * np.exp(-np.asarray(t, float)/tauM)


def maxwell_time(eta, G0=G_SHEAR):
    """Maxwell relaxation time tauM = eta/G0 (s)."""
    return eta / G0
