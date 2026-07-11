#!/usr/bin/env python3
r"""
make_figure.py  --  publication figure for the dike-focusing model (TuLIP 2.5).

Design (revised): the figure is built around a single controlled comparison that
anchors the meaning of the focusing radius R_f.

  (a) Reservoir in an ISOTROPIC ambient stress (K0=1, sigma_tect=0, so the ambient
      DIFFERENTIAL stress Dsig_bg = 0). Meridional section: colour = reservoir
      differential stress (sigma1-sigma3); white curves = sigma1 trajectories =
      dike paths. With no ambient differential stress there is no competing length
      scale, every trajectory is reoriented toward the source, and the focusing
      radius is unbounded (R_f -> infinity). This is the reference case that shows
      Dsig_bg is what sets the scale, NOT an "opposing pressure".

  (b) SAME reservoir, SAME overpressure, but now with a finite ambient differential
      stress (K0=0.98 -> Dsig_bg = |1-K0| rho g z). The reservoir only wins out to
      the cross-over radius where its differential stress (3/2) dP a^3/R^3 equals
      Dsig_bg. Dikes seeded inside R_f are captured/steered to the summit; dikes
      seeded outside R_f straighten to the regional vertical. R_f is marked.

  (c) Derived focusing-radius law R_f/a = (3/2 * dP/Dsig_bg)^{1/3}. The x-axis is
      the ratio of reservoir overpressure to AMBIENT DIFFERENTIAL STRESS (sigma1 -
      sigma3), not to a pressure. Brackets the ad-hoc "3x" heuristic and the
      Pansino & Taisne (2019) 2-10x range.

  (d) Viscoelastic shut-off. Fixed-volume relaxation gives dP(t)=dP0 G(t)/G0, hence
      A_f(t)/A_f0 = (G(t)/G0)^{2/3}. Curves for Maxwell and Burgers/extended-Burgers
      crust and sediment presets; the transient (anelastic) knee and the steady
      (viscous) shut-off are annotated.

The Eshelby/CLVD source-shape diagnostic has been moved out of the science figure
(it is a geodetic/seismic moment-tensor observable, not part of the overpressure ->
stress -> focusing -> area causal chain). It now lives only in the verification /
benchmark figures (verify_mogi_eshelby.py, benchmark_freesurface.py).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import focusing as fo
import rheology as rh

plt.rcParams.update({
    'font.size': 9, 'axes.linewidth': 0.8, 'axes.labelsize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
YEAR = rh.YEAR

# common reservoir geometry / loading for the two section panels
a, d, dP = 1000., 3000., 15e6
z0_seed = -8000.                       # seed depth for dike trajectories
seed_r0 = np.linspace(150., 6500., 15)
th = np.linspace(0, 2*np.pi, 120)      # reservoir outline

# shared meridional grid + colour scale (reservoir differential stress)
rr = np.linspace(1., 9000., 340)
zz = np.linspace(-9000., 0., 320)
Rg, Zg = np.meshgrid(rr, zz)
R_src = np.sqrt(Rg**2 + (Zg + d)**2)
Dsig_res = fo.reservoir_differential_stress(np.maximum(R_src, a), dP, a)  # Pa
LOGV = (np.log10(np.abs(Dsig_res) / 1e6), -1.0, 2.0)   # (field, vmin, vmax) in MPa


def section_panel(ax, K0, sig_t, title, mark_Rf):
    """Draw one meridional section: reservoir differential stress + sigma1 dikes."""
    pc = ax.pcolormesh(Rg / 1e3, Zg / 1e3, LOGV[0], cmap='viridis',
                       shading='auto', vmin=LOGV[1], vmax=LOGV[2])
    # reservoir body
    ax.fill(a * np.cos(th) / 1e3, (-d + a * np.sin(th)) / 1e3,
            color='crimson', ec='k', lw=0.8, zorder=6)
    # sigma1 trajectories = dike paths
    for r0 in seed_r0:
        out = fo.trace_dike(r0, z0_seed, dP, a, d, K0=K0, sigma_tect=sig_t, ds=120.)
        col = 'w' if out['captured_reservoir'] else '0.75'
        ax.plot(out['r'] / 1e3, out['z'] / 1e3, color=col, lw=0.8, alpha=0.9)
    if mark_Rf:
        # shell-averaged Dsig_bg (= value at source depth d for a linear background)
        Dsig_bg = float(fo.background_differential_stress_shellavg(d, K0=K0, sigma_tect=sig_t))
        Rc = fo.focusing_radius_closed_form(dP, a, Dsig_bg)         # reorientation radius
        # streamline capture radius from sigma1 trajectories (deep-seed, saturated)
        Rcap, *_ = fo.focusing_radius_trajectory(dP, a, d, z0_seed, K0=K0,
                                                 sigma_tect=sig_t, n_seed=120)
        ax.axvspan(0, Rc / 1e3, color='w', alpha=0.10)
        ax.axvline(Rc / 1e3, color='w', ls='--', lw=1.2)
        ax.text(Rc / 1e3 + 0.12, -0.5, r'$R_c$', color='w', fontsize=11, fontweight='bold')
        ax.axvline(Rcap / 1e3, color='w', ls=':', lw=1.1)
        ax.text(Rcap / 1e3 + 0.12, -1.15, r'$R_{cap}$', color='w', fontsize=9)
        # viscoelastic shrinkage: R_c(t) = R_c (G(t)/G0)^{1/3}. Mark at G/G0 = e^-1, e^-2
        for gfac, lab in [(np.exp(-1.0), r'$t=\tau_M$'), (np.exp(-2.0), r'$t=2\tau_M$')]:
            Rt = Rc * gfac ** (1.0 / 3.0)
            ax.axvline(Rt / 1e3, color='gold', ls='-', lw=0.9, alpha=0.9)
            ax.text(Rt / 1e3 - 0.05, -8.4, lab, color='gold', fontsize=7,
                    ha='right', rotation=90, va='bottom')
        ax.text(0.97, 0.03,
                rf'$\Delta\sigma_{{bg}}={Dsig_bg/1e6:.2f}$ MPa (shell-avg)'
                f'\n' rf'$R_c={Rc/1e3:.2f}$ km$={Rc/a:.1f}\,a$'
                f'\n' rf'$R_{{cap}}={Rcap/1e3:.2f}$ km ($\approx{Rcap/Rc:.2f}\,R_c$)',
                transform=ax.transAxes, ha='right', va='bottom', color='w',
                fontsize=7.6,
                bbox=dict(boxstyle='round', fc='k', ec='none', alpha=0.4))
    else:
        ax.text(0.97, 0.03,
                r'$\Delta\sigma_{bg}=0$' '\n(no length scale:' '\n' r'$R_f\!\to\!\infty$)',
                transform=ax.transAxes, ha='right', va='bottom', color='w',
                fontsize=8.0,
                bbox=dict(boxstyle='round', fc='k', ec='none', alpha=0.35))
    ax.set_xlim(0, 9); ax.set_ylim(-9, 0)
    ax.set_xlabel('radial distance [km]'); ax.set_ylabel('depth [km]')
    ax.set_title(title, fontsize=10, loc='left')
    return pc


fig = plt.figure(figsize=(11, 8.8))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.27)

# (a) isotropic ambient stress: Dsig_bg = 0, focusing unbounded
axa = fig.add_subplot(gs[0, 0])
pca = section_panel(axa, K0=1.0, sig_t=0.0,
                    title=r'(a) isotropic ambient ($\Delta\sigma_{bg}=0$): all dikes focus',
                    mark_Rf=False)

# (b) finite ambient differential stress: finite R_f
axb = fig.add_subplot(gs[0, 1])
pcb = section_panel(axb, K0=0.98, sig_t=0.0,
                    title=r'(b) finite $\Delta\sigma_{bg}$: focusing out to $R_f$',
                    mark_Rf=True)
cb = fig.colorbar(pcb, ax=[axa, axb], pad=0.015, fraction=0.046)
cb.set_label(r'$\log_{10}\,$ reservoir $(\sigma_1-\sigma_3)$ [MPa]')

# (c) derived focusing-radius law
axc = fig.add_subplot(gs[1, 0])
ratio = np.logspace(-0.5, 2.2, 200)                 # dP / Dsig_bg
Rf_over_a = (1.5 * ratio) ** (1 / 3)
axc.loglog(ratio, Rf_over_a, 'k-', lw=1.8,
           label=r'$R_f/a=\left(\frac{3}{2}\,\Delta P/\Delta\sigma_{bg}\right)^{1/3}$')
axc.axhspan(2, 10, color='tab:blue', alpha=0.12, label='Pansino & Taisne (2019), 2--10$\\times$')
axc.axhline(3, color='tab:red', ls=':', lw=1.2, label='ad-hoc "3$\\times$" heuristic')
# mark the panel-(b) operating point
ratio_b = dP / float(fo.background_differential_stress(d, K0=0.98, sigma_tect=0.0))
axc.plot(ratio_b, (1.5 * ratio_b) ** (1 / 3), 'o', color='crimson', ms=7,
         zorder=5, label='panel (b) state')
axc.set_xlabel(r'$\Delta P\,/\,\Delta\sigma_{bg}$  (overpressure / ambient differential stress)')
axc.set_ylabel(r'focusing radius $R_f/a$')
axc.set_title('(c) derived focusing-radius law', fontsize=10, loc='left')
axc.legend(fontsize=7.5, loc='upper left')
axc.grid(True, which='both', alpha=0.25)

# (d) viscoelastic shut-off of the focusing area
#     The shut-off time is set by the HOST viscosity, which spans orders of
#     magnitude. Bulk crystalline crust has eta~1e19-1e23 Pa s (Burgmann & Dresen
#     2008), giving tau_M = eta/G0 of decades to >1e5 yr; only the thin heated
#     contact aureole immediately against the sill reaches eta~1e18 (tau_M~yr).
#     Plotting the 1e18 aureole value as if it were "the crust" makes focusing
#     look like it dies in ~2 yr, which is spurious -- it is a lower bound confined
#     to a metre-to-decametre-thick skin. The physically relevant comparison is the
#     LIP sill magmatic lifetime (emplacement + solidification), ~1e2-1e5 yr.
axd = fig.add_subplot(gs[1, 1])
t = np.logspace(np.log10(0.05 * YEAR), np.log10(1e5 * YEAR), 500)
# LIP sill magmatic lifetime band (emplacement -> solidification of a sill complex)
axd.axvspan(1e2, 1e5, color='tab:purple', alpha=0.08, zorder=0)
axd.text(3.2e3, 0.90, 'LIP sill magmatic\nlifetime ($10^2$--$10^5$ yr)',
         color='tab:purple', fontsize=7.0, ha='center', va='top')
# PRIMARY: bulk crust as a single-time Maxwell reference (eta=1e21 Pa s)
mx = rh.MaxwellRheology(G0=20e9, etaM=1e21)
axd.semilogx(t / YEAR, (mx.G(t) / mx.G0) ** (2 / 3), 'k-', lw=1.8,
             label=rf'Maxwell, bulk crust ($\eta=10^{{21}}$, $\tau_M\approx{mx.tauM/YEAR/1e3:.1f}$ kyr)')
# Burgers crust: transient (anelastic) knee + steady (viscous) shut-off
for name, model, c in [('crustal rock (Burgers)', rh.crustal_rock_burgers(), 'tab:brown'),
                       ('sediment (Burgers)', rh.sediment_burgers(), 'tab:green')]:
    axd.semilogx(t / YEAR, (model.G(t) / model.G0) ** (2 / 3), lw=1.6,
                 color=c, label=name)
ext = rh.crustal_extended_burgers()
g_ext = np.clip(np.asarray(ext.G(t)) / ext.G0, 0.0, None)   # ODE tail can dip <0
axd.semilogx(t / YEAR, g_ext ** (2 / 3), '--',
             color='0.35', lw=1.4, label='crust (extended Burgers)')
# FAST BRACKET: hot contact aureole -- explicitly a thin heated skin, not bulk crust
hot = rh.hot_aureole_burgers()
axd.semilogx(t / YEAR, (np.asarray(hot.G(t)) / hot.G0) ** (2 / 3), ':', lw=1.5,
             color='tab:red', label=r'hot aureole (thin skin, $\eta\!\sim\!10^{18}$)')
axd.annotate('transient\n(anelastic) knee', xy=(20, 0.80), xytext=(0.3, 0.55),
             fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
axd.annotate('steady (viscous)\nshut-off (bulk crust)', xy=(2e3, 0.35), xytext=(2e3, 0.62),
             fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', lw=0.8))
axd.set_xlabel('time since pressurization [yr]')
axd.set_ylabel(r'normalized focusing area  $A_f(t)/A_{f,0}=(G(t)/G_0)^{2/3}$')
axd.set_title('(d) viscoelastic shut-off of focusing', fontsize=10, loc='left')
axd.legend(fontsize=7.0, loc='lower left')
axd.set_ylim(0, 1.02)
axd.grid(True, which='both', alpha=0.25)

os.makedirs('figures', exist_ok=True)
fig.savefig('figures/dike_focusing.png')
fig.savefig('figures/dike_focusing.pdf')

Dsig_bg_b = float(fo.background_differential_stress(d, K0=0.98, sigma_tect=0.0))
Rf_b = fo.focusing_radius_closed_form(dP, a, Dsig_bg_b)
print("wrote figures/dike_focusing.{png,pdf}")
print(f"  panel(a): Dsig_bg=0  -> R_f -> infinity (all dikes focus)")
print(f"  panel(b): dP={dP/1e6:.0f} MPa, Dsig_bg={Dsig_bg_b/1e6:.2f} MPa "
      f"-> R_f={Rf_b:.0f} m = {Rf_b/a:.2f} a")
print(f"  panel(d) characteristic relaxation timescales (A_f half-life = 1.5*ln2*tau_M):")
print(f"    bulk-crust Maxwell (eta=1e21):  tau_M={mx.tauM/YEAR:.0f} yr, A_f half-life={1.5*np.log(2)*mx.tauM/YEAR:.0f} yr")
print(f"    crustal rock (Burgers): transient knee ~{rh.crustal_rock_burgers().times()['tau_fast']/YEAR:.0f} yr,"
      f" steady ~{rh.crustal_rock_burgers().times()['tau_slow']/YEAR:.0f} yr")
print(f"    hot aureole (fast bracket, thin skin): steady ~{hot.times()['tau_slow']/YEAR:.1f} yr")
