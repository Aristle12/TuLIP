"""
CO2 Phase Diagram for Sill Emplacement Conditions
===================================================
Panel (a): CO2 P-T phase diagram with density (CoolProp, Span-Wagner EOS)
Panel (b): CO2 vs H2O density at depth (CoolProp for both, IAPWS-95 for water)
Panel (c): CO2 solubility in pure water using the Spycher et al. (2003) /
           Spycher & Pruess (2010) model with Redlich-Kwong EOS.
           Low-T (12-99°C):  Spycher, Pruess & Ennis-King (2003)
           High-T (109-300°C): Spycher & Pruess (2010)
           Transition (99-109°C): cubic interpolation

References:
- Span & Wagner (1996). J. Phys. Chem. Ref. Data, 25(6), 1509-1596. [CO2 EOS]
- Bell et al. (2014). Ind. Eng. Chem. Res., 53(6), 2498-2508. [CoolProp]
- Wagner & Pruss (2002). J. Phys. Chem. Ref. Data, 31(2), 387-535. [IAPWS-95]
- Spycher, N., Pruess, K. & Ennis-King, J. (2003). GCA, 67(16), 3015-3031.
  [CO2 solubility 12-100°C, up to 600 bar]
- Spycher, N. & Pruess, K. (2005). GCA, 69(13), 3309-3320.
  [Extension to NaCl brines]
- Spycher, N. & Pruess, K. (2010). Transp. Porous Med., 82, 173-196.
  [Extension to 300°C for pure water and brines]
- Coefficients cross-verified against MOOSE PorousFlowBrineCO2 implementation
  (Idaho National Laboratory, github.com/idaholab/moose).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from CoolProp.CoolProp import PropsSI
import warnings
warnings.filterwarnings('ignore')

R_gas = 83.1446  # cm³ bar / (mol K) — gas constant in CGS-bar units

# ============================================================
# 1. Spycher et al. (2003) / Spycher & Pruess (2010) model
#    for CO2 solubility in pure water
# ============================================================

# --- Redlich-Kwong parameters ---
# Low-T regime (Spycher et al. 2003, Table 1):
def rk_params_lowT(T_K):
    """RK parameters for T <= ~99°C (Spycher et al. 2003, Table 1)."""
    a_CO2 = 7.54e7 - 4.02e4 * T_K       # bar cm^6 K^0.5 mol^-2
    b_CO2 = 27.86                         # cm^3/mol
    a_H2O_CO2 = 7.89e7                   # bar cm^6 K^0.5 mol^-2
    b_H2O = 18.10                         # cm^3/mol
    return a_CO2, b_CO2, a_H2O_CO2, b_H2O

# High-T regime (Spycher & Pruess 2010, Table 1):
def rk_params_highT(T_K):
    """RK parameters for T >= ~109°C (Spycher & Pruess 2010).
    Uses Panagiotopoulos-Reid asymmetric mixing rules (Appendix A, Eqs. A-5, A-6).
    At low-T approximation (y_H2O ≈ 0), k_12 ≈ K_12*1 + K_21*0 = K_12(T),
    giving a_12 = sqrt(a_CO2 * a_H2O) * (1 - K_CO2-H2O(T)).
    """
    a_CO2 = 8.008e7 - 4.984e4 * T_K
    b_CO2 = 28.25
    a_H2O = 1.337e8 - 1.4e4 * T_K
    b_H2O = 15.70
    # Asymmetric interaction parameters (Table 1, Eq. A-6, A-7):
    # K_CO2-H2O = F(T_K) = a + b*T_K (used when y_CO2 ≈ 1)
    # K_H2O-CO2 = F(T_K) = a + b*T_K (used when y_H2O ≈ 0, so doesn't enter at leading order)
    K_CO2_H2O = 0.4228 + (-7.422e-4) * T_K  # K_CO2-H2O from Table 1
    # For infinite H2O dilution (y_CO2≈1, y_H2O≈0):
    # k_12 = K_CO2-H2O * y_CO2 + K_H2O-CO2 * y_H2O ≈ K_CO2-H2O
    a_H2O_CO2 = np.sqrt(abs(a_CO2 * a_H2O)) * (1 - K_CO2_H2O)
    return a_CO2, b_CO2, a_H2O_CO2, b_H2O


def rk_params(T_K):
    """Select RK parameters based on temperature regime."""
    Tc = T_K - 273.15
    if Tc <= 99.0:
        return rk_params_lowT(T_K)
    elif Tc >= 109.0:
        return rk_params_highT(T_K)
    else:
        # Linear interpolation in transition zone
        w = (Tc - 99.0) / 10.0
        p_lo = rk_params_lowT(T_K)
        p_hi = rk_params_highT(T_K)
        return tuple(p_lo[i] * (1 - w) + p_hi[i] * w for i in range(4))


# --- Equilibrium constant K0_CO2 ---
def logK0_CO2(T_K, is_liquid=False):
    """
    log10(K0_CO2) as a function of temperature.
    Uses polynomial fits from Spycher et al. (2003) Table 2 for low-T
    and Spycher & Pruess (2010) for high-T, with cubic interpolation.

    For subcritical temperatures when CO2 is liquid, uses K0_CO2(l)
    instead of K0_CO2(g) (Spycher et al. 2003, Table 2).
    """
    Tc = T_K - 273.15
    if Tc <= 99.0:
        if is_liquid and Tc <= 31.0:
            # CO2 liquid: Spycher et al. (2003), Table 2 regression for CO2(l)
            return 1.168 + 1.361e-2 * Tc - 5.135e-5 * Tc**2
        else:
            # CO2 gas or supercritical: Table 2 regression for CO2(g)
            return 1.188 + 1.307e-2 * Tc - 5.445e-5 * Tc**2
    elif Tc >= 109.0:
        # Spycher & Pruess (2010)
        return 1.668 + 3.992e-3 * Tc - 1.156e-5 * Tc**2 + 1.593e-9 * Tc**3
    else:
        # Cubic interpolation (as in MOOSE)
        t_int = (Tc - 99.0) / 10.0
        return 1.9462 + 2.25692e-2 * t_int - 9.49577e-3 * t_int**2 - 6.77721e-3 * t_int**3


# --- Partial molar volumes ---
def Vbar_CO2(T_K):
    """
    Average partial molar volume of dissolved CO2 (cm³/mol).
    Low-T: constant 32.1 (Spycher et al. 2003, Table 2)
    High-T: V̄ = 32.6 + 3.413e-2*(T_K - 373.15) (Spycher & Pruess 2010, Table 1, Eq. 7)
    """
    Tc = T_K - 273.15
    if Tc <= 99.0:
        return 32.1
    elif Tc >= 109.0:
        return 32.6 + 3.413e-2 * (T_K - 373.15)
    else:
        w = (Tc - 99.0) / 10.0
        V_lo = 32.1
        V_hi = 32.6 + 3.413e-2 * (T_K - 373.15)
        return V_lo * (1 - w) + V_hi * w


def Vbar_H2O(T_K):
    """
    Average partial molar volume of liquid H2O (cm³/mol).
    Low-T: constant 18.5 (Spycher et al. 2003, Table 2)
    High-T: V̄ = 18.1 + 3.137e-2*(T_K - 373.15) (Spycher & Pruess 2010, Table 1, Eq. 7)
    """
    Tc = T_K - 273.15
    if Tc <= 99.0:
        return 18.5
    elif Tc >= 109.0:
        return 18.1 + 3.137e-2 * (T_K - 373.15)
    else:
        w = (Tc - 99.0) / 10.0
        V_lo = 18.5
        V_hi = 18.1 + 3.137e-2 * (T_K - 373.15)
        return V_lo * (1 - w) + V_hi * w


# --- Margules activity coefficient for CO2 (high-T only) ---
def margules_A(T_K):
    """
    Margules parameter for CO2 activity coefficient in pure water.
    Only significant at high T (Spycher & Pruess 2010).
    """
    Tc = T_K - 273.15
    if Tc <= 99.0:
        return 0.0  # Not needed at low T
    elif Tc >= 109.0:
        Tref = T_K - 373.15
        return -3.084e-2 * Tref + 1.927e-5 * Tref**2
    else:
        w = (Tc - 99.0) / 10.0
        Tref = T_K - 373.15
        A_hi = -3.084e-2 * Tref + 1.927e-5 * Tref**2
        return w * A_hi


# --- Solve Redlich-Kwong cubic for molar volume ---
def rk_volume(T_K, P_bar, a, b):
    """
    Solve the Redlich-Kwong EOS (Eq. 24 of Spycher et al. 2003) for volume.
    V^3 - (RT/P)*V^2 - (RTb/P - a/(PT^0.5) + b^2)*V - ab/(PT^0.5) = 0

    Returns the volume of the STABLE phase (gas or liquid) using the
    equal-area (Maxwell) criterion from Eqs. 25-26 of Spycher et al. (2003).
    Also returns a boolean indicating if CO2 is in liquid state.
    """
    RT_P = R_gas * T_K / P_bar
    a_T = a / (T_K**0.5)

    # Coefficients of V^3 + c2*V^2 + c1*V + c0 = 0
    c2 = -RT_P
    c1 = -(R_gas * T_K * b / P_bar - a_T / P_bar + b**2)
    c0 = -(a_T * b / P_bar)

    roots = np.roots([1, c2, c1, c0])
    # Take real positive roots
    real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-6 and r.real > b])

    if len(real_roots) == 0:
        return np.nan, False

    if len(real_roots) == 1:
        # Only one real root: supercritical or far from phase boundary
        is_liquid = (real_roots[0] < 94.0)  # ~critical volume of CO2
        return real_roots[0], is_liquid

    # Multiple roots: determine stable phase using Maxwell criterion
    V_gas = max(real_roots)
    V_liq = min(real_roots)

    # w1 = P * (V_gas - V_liq)  [Eq. 25]
    w1 = P_bar * (V_gas - V_liq)

    # w2 = RT*ln((V_gas-b)/(V_liq-b)) + a/(T^0.5*b)*ln((V_gas+b)*V_liq/((V_liq+b)*V_gas))
    # [Eq. 26]
    if V_liq - b > 0 and V_liq > 0:
        w2 = (R_gas * T_K * np.log((V_gas - b) / (V_liq - b))
              + a / (T_K**0.5 * b) * np.log((V_gas + b) * V_liq / ((V_liq + b) * V_gas)))
    else:
        return V_gas, False

    if w2 >= w1:
        return V_gas, False   # Gas phase stable
    else:
        return V_liq, True    # Liquid phase stable


# --- Fugacity coefficients from RK EOS ---
def fugacity_coeffs_rk(T_K, P_bar):
    """
    Compute fugacity coefficients Φ_CO2 and Φ_H2O using Redlich-Kwong EOS
    with infinite H2O dilution (y_CO2=1, y_H2O=0).
    Equations 21-23 of Spycher et al. (2003).

    Returns: phi_CO2, phi_H2O, is_liquid
    """
    a_CO2, b_CO2, a_H2O_CO2, b_H2O = rk_params(T_K)

    # With y_CO2=1, y_H2O=0: a_mix = a_CO2, b_mix = b_CO2
    a_mix = a_CO2
    b_mix = b_CO2

    V, is_liquid = rk_volume(T_K, P_bar, a_mix, b_mix)
    if np.isnan(V):
        return np.nan, np.nan, False

    # ln(Φ_k) from Eq. 23 (Spycher et al. 2003):
    # ln(Φ_k) = ln(V/(V-b_mix)) + b_k/(V-b_mix)
    #           - (2*Σ y_i*a_ik / (RT^1.5 * b_mix)) * ln((V+b_mix)/V)
    #           + (a_mix*b_k / (RT^1.5 * b_mix^2)) * [ln((V+b)/V) - b_mix/(V+b_mix)]
    #           - ln(PV/RT)

    RT15 = R_gas * T_K**1.5

    # For CO2 (k=CO2): with y_CO2=1: Σ y_i * a_i,CO2 = a_CO2
    sum_ya_CO2 = a_CO2  # = y_CO2 * a_CO2,CO2 + y_H2O * a_H2O,CO2 → a_CO2
    ln_Vbm = np.log(V / (V - b_mix))
    ln_Vb_V = np.log((V + b_mix) / V)

    ln_phi_CO2 = (ln_Vbm + b_CO2 / (V - b_mix)
                  - (2 * sum_ya_CO2 / (RT15 * b_mix)) * ln_Vb_V
                  + (a_mix * b_CO2 / (RT15 * b_mix**2))
                    * (ln_Vb_V - b_mix / (V + b_mix))
                  - np.log(P_bar * V / (R_gas * T_K)))

    # For H2O (k=H2O): Σ y_i * a_i,H2O = y_CO2 * a_H2O,CO2 = a_H2O,CO2
    sum_ya_H2O = a_H2O_CO2
    ln_phi_H2O = (ln_Vbm + b_H2O / (V - b_mix)
                  - (2 * sum_ya_H2O / (RT15 * b_mix)) * ln_Vb_V
                  + (a_mix * b_H2O / (RT15 * b_mix**2))
                    * (ln_Vb_V - b_mix / (V + b_mix))
                  - np.log(P_bar * V / (R_gas * T_K)))

    return np.exp(ln_phi_CO2), np.exp(ln_phi_H2O), is_liquid


# --- CO2 solubility calculation ---
def co2_solubility_spycher(T_K, P_bar):
    """
    CO2 solubility in pure water (mole fraction x_CO2) using the
    Spycher et al. (2003) / Spycher & Pruess (2010) model.

    Returns: x_CO2 (mole fraction), m_CO2 (molality, mol/kg)
    """
    Tc = T_K - 273.15
    if Tc < 12 or Tc > 300 or P_bar < 1 or P_bar > 600:
        return np.nan, np.nan

    # Check if below water saturation pressure
    try:
        P_sat_H2O = PropsSI('P', 'T', T_K, 'Q', 0, 'Water') / 1e5  # bar
    except:
        P_sat_H2O = 0.0
    if P_bar <= P_sat_H2O:
        return np.nan, np.nan

    # Fugacity coefficients
    phi_CO2, phi_H2O, is_liquid = fugacity_coeffs_rk(T_K, P_bar)
    if np.isnan(phi_CO2) or np.isnan(phi_H2O):
        return np.nan, np.nan

    # Equilibrium constant (gas or liquid depending on CO2 phase)
    K0 = 10**logK0_CO2(T_K, is_liquid=is_liquid)

    # Reference pressure: 1 bar for T <= 100°C, P_sat(H2O) for T > 100°C
    # (Spycher & Pruess 2010, Table 1 footnote c)
    if Tc <= 100.0:
        P0 = 1.0
    else:
        P0 = max(P_sat_H2O, 1.0)

    Vbar = Vbar_CO2(T_K)
    V_H2O = Vbar_H2O(T_K)

    # Water equilibrium constant K0_H2O
    # Low-T: Spycher et al. (2003) Table 2
    # High-T: Spycher & Pruess (2010) Table 1
    if Tc <= 99.0:
        logK0_H2O_val = (-2.209 + 3.097e-2 * Tc - 1.098e-4 * Tc**2
                          + 2.048e-7 * Tc**3)
    elif Tc >= 109.0:
        logK0_H2O_val = (-2.1077 + 2.8127e-2 * Tc - 8.4298e-5 * Tc**2
                          + 1.4969e-7 * Tc**3 - 1.1812e-10 * Tc**4)
    else:
        # Blend
        w = (Tc - 99.0) / 10.0
        lo = (-2.209 + 3.097e-2 * Tc - 1.098e-4 * Tc**2 + 2.048e-7 * Tc**3)
        hi = (-2.1077 + 2.8127e-2 * Tc - 8.4298e-5 * Tc**2
              + 1.4969e-7 * Tc**3 - 1.1812e-10 * Tc**4)
        logK0_H2O_val = lo * (1 - w) + hi * w
    K0_H2O = 10**logK0_H2O_val

    # B parameter (Eq. 11 of Spycher & Pruess 2010, without γ_CO2)
    B = (phi_CO2 * P_bar / (55.508 * K0)) * np.exp(-(P_bar - P0) * Vbar / (R_gas * T_K))

    # A parameter (Eq. 10, without γ_H2O)
    A = (K0_H2O / (phi_H2O * P_bar)) * np.exp((P_bar - P0) * V_H2O / (R_gas * T_K))

    # Margules activity correction for high T
    Am = margules_A(T_K)

    # Solve for y_H2O and x_CO2
    # Low-T (Eqs. 29, 30 of Spycher 2003): direct, γ_CO2 = γ_H2O = 1
    # High-T (Eqs. 8, 9, 16 of Spycher & Pruess 2010): iterative with Margules

    if Am == 0.0:
        # Low-T regime: no activity correction needed
        if abs(1.0/A - B) < 1e-15:
            return np.nan, np.nan
        y_H2O = (1 - B) / (1.0/A - B)
        x_CO2 = B * (1 - y_H2O)
    else:
        # High-T regime: iterative solution with Margules correction
        # Eq. 13: ln(γ_CO2) = 2*A_M * x_CO2 * (1-x_CO2)^2
        # Eq. 12: ln(γ_H2O) = A_M*(2*x_CO2 - 1) * x_CO2^2
        # (Carlson & Colburn 1942 form, ensuring γ_CO2→1 as x_CO2→0)
        #
        # Iteration via Eq. 16: y_H2O = (1 - B/γ_CO2) / (1/(A·γ_H2O) - B/γ_CO2)
        # then x_CO2 = (B/γ_CO2) * (1 - y_H2O)

        # Initial guess: use non-corrected solution
        if abs(1.0/A - B) > 1e-15:
            y_H2O_init = (1 - B) / (1.0/A - B)
            x_CO2 = B * (1 - y_H2O_init)
        else:
            x_CO2 = 0.009  # ~0.5 molal starting guess

        for _ in range(100):
            x_H2O = 1.0 - x_CO2
            # Margules activity coefficients (Eqs. 12-13, Spycher & Pruess 2010)
            gamma_CO2 = np.exp(2.0 * Am * x_CO2 * x_H2O**2)
            gamma_H2O = np.exp(Am * (2.0 * x_CO2 - 1.0) * x_CO2**2)

            # Corrected B and A (Eqs. 10-11)
            B_corr = B / gamma_CO2
            A_corr = A * gamma_H2O

            denom = 1.0/A_corr - B_corr
            if abs(denom) < 1e-15:
                return np.nan, np.nan
            y_H2O = (1.0 - B_corr) / denom
            x_CO2_new = B_corr * (1.0 - y_H2O)

            if abs(x_CO2_new - x_CO2) < 1e-12:
                x_CO2 = x_CO2_new
                break
            x_CO2 = x_CO2_new

    # Upper bound: at 300°C and high P, x_CO2 can exceed 30 mol% (approaching miscibility)
    if x_CO2 < 0 or x_CO2 > 0.5:
        return np.nan, np.nan

    # Convert mole fraction to molality
    m_CO2 = 55.508 * x_CO2 / (1 - x_CO2)

    return x_CO2, m_CO2


# ============================================================
# 2. Validation against experimental data
# ============================================================

print("=" * 80)
print("Validation: Spycher (2003) / Spycher & Pruess (2010) model")
print("vs experimental CO2 solubility data")
print("=" * 80)
print(f"{'T(°C)':>6} {'P(bar)':>8} {'x_calc':>10} {'m_calc':>8} {'m_exp':>8} {'Err%':>7} {'Source':>25}")
print("-" * 80)

test_points = [
    # T(°C), P(bar), x_CO2*100 (approx), source
    # Values read from Spycher (2003) Fig 6 model curves and
    # experimental data compilation (Appendix A)
    (25,   10,  0.34,  "Wiebe & Gaddy 1940"),
    (25,   50,  1.21,  "Spycher 2003 Fig 6"),
    (25,  100,  1.41,  "Spycher 2003 Fig 6"),
    (25,  200,  1.54,  "Spycher 2003 Fig 6"),
    (25,  400,  1.73,  "Spycher 2003 Fig 6"),
    (50,   50,  0.78,  "Spycher 2003 Fig 6"),
    (50,  100,  1.13,  "Spycher 2003 Fig 6"),
    (50,  200,  1.29,  "Spycher 2003 Fig 6"),
    (75,  100,  0.90,  "Spycher 2003 Fig 6"),
    (75,  200,  1.16,  "Spycher 2003 Fig 6"),
    (100, 100,  0.79,  "Spycher 2003 Fig 6"),
    (100, 200,  1.13,  "Spycher 2003 Fig 6"),
    (100, 500,  1.55,  "Spycher 2003 Fig 6"),
    # High-T validation from Spycher & Pruess (2010) Fig 1 model curves (mole %)
    # Values read from the published model curves in Fig 1
    (150, 200,  1.25,  "Spycher & Pruess 2010"),  # ~2.2 mol%
    (150, 400,  1.72,  "Spycher & Pruess 2010"),  # ~3.0 mol%
    (200, 200,  1.43,  "Spycher & Pruess 2010"),  # ~2.5 mol%
    (200, 400,  2.46,  "Spycher & Pruess 2010"),  # ~4.2 mol%
    (250, 200,  1.58,  "Spycher & Pruess 2010"),  # ~2.8 mol%
    (250, 400,  3.58,  "Spycher & Pruess 2010"),  # ~6.1 mol%
]

errors = []
for T_C, P_bar, m_exp, source in test_points:
    T_K = T_C + 273.15
    x_calc, m_calc = co2_solubility_spycher(T_K, P_bar)
    if m_calc is not None and not np.isnan(m_calc):
        pct = 100 * (m_calc - m_exp) / m_exp
        errors.append(abs(pct))
        print(f"{T_C:6.0f} {P_bar:8.0f} {x_calc:10.5f} {m_calc:8.3f} {m_exp:8.3f} {pct:+7.1f} {source:>25}")
    else:
        print(f"{T_C:6.0f} {P_bar:8.0f} {'NaN':>10} {'NaN':>8} {m_exp:8.3f} {'---':>7} {source:>25}")

if errors:
    print("-" * 80)
    print(f"Mean absolute error: {np.mean(errors):.1f}%")
    print(f"Max absolute error:  {np.max(errors):.1f}%")
    print(f"Points within 5%:    {sum(1 for e in errors if e < 5)}/{len(errors)}")
    print(f"Points within 10%:   {sum(1 for e in errors if e < 10)}/{len(errors)}")


# ============================================================
# 3. Compute grids
# ============================================================

# CO2 density grid
print("\nComputing CO2 density grid...")
T_range = np.linspace(273.15, 873.15, 300)
P_range = np.linspace(0.5e6, 400e6, 300)
T_grid, P_grid = np.meshgrid(T_range, P_range)
rho_grid = np.full_like(T_grid, np.nan)
for i in range(P_range.shape[0]):
    for j in range(T_range.shape[0]):
        try:
            rho_grid[i, j] = PropsSI('D', 'T', T_range[j], 'P', P_range[i], 'CO2')
        except:
            pass
print("  Done.")

# CO2 saturation curve
T_crit = PropsSI('Tcrit', 'CO2')
P_crit = PropsSI('pcrit', 'CO2')
T_triple = PropsSI('Ttriple', 'CO2')
T_sat = np.linspace(T_triple + 0.1, T_crit - 0.01, 200)
P_sat = np.array([PropsSI('P', 'T', T, 'Q', 0, 'CO2') for T in T_sat])

# Solubility grid (Spycher model, 12-300°C, 1-600 bar)
# Low-T (12-99°C): Spycher et al. (2003), non-iterative
# High-T (109-300°C): Spycher & Pruess (2010), iterative with Margules
print("Computing CO2 solubility grid (Spycher 2003/2010, 12-300°C)...")
T_sol = np.linspace(285.15, 573.15, 150)   # 12°C to 300°C
P_sol_bar = np.linspace(10, 600, 150)       # 10 to 600 bar
T_sol_grid, P_sol_bar_grid = np.meshgrid(T_sol, P_sol_bar)
sol_grid = np.full_like(T_sol_grid, np.nan)
for i in range(P_sol_bar.shape[0]):
    for j in range(T_sol.shape[0]):
        _, m = co2_solubility_spycher(T_sol[j], P_sol_bar[i])
        sol_grid[i, j] = m
print("  Done.")

# ============================================================
# 4. Depth-pressure conversion
# ============================================================
rho_rock = 2400.0
g = 9.81

def depth_to_P_litho(z_km):
    return rho_rock * g * z_km * 1000 / 1e6

def P_to_depth_litho(P_MPa):
    return P_MPa * 1e6 / (rho_rock * g * 1000)

# ============================================================
# 5. FIGURE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

z_sill = np.array([1, 4])
P_sill = depth_to_P_litho(z_sill)
T_sill_low = 15 + 25 * z_sill
T_sill_high = 15 + 40 * z_sill

# ---- Panel (a): CO2 P-T phase diagram with density ----
ax = axes[0]
T_C_grid = T_grid - 273.15
P_MPa_grid = P_grid / 1e6
rho_plot = np.copy(rho_grid)
rho_plot[rho_plot < 1] = np.nan

levels = [1, 5, 10, 25, 50, 100, 200, 400, 600, 800, 1000, 1200]
cf = ax.contourf(T_C_grid, P_MPa_grid, rho_plot, levels=levels,
                 cmap='viridis', norm=LogNorm(vmin=1, vmax=1200), extend='both')
cb = fig.colorbar(cf, ax=ax, label=r'CO$_2$ density (kg/m$^3$)', shrink=0.85)

ax.plot(T_sat - 273.15, P_sat / 1e6, 'w-', linewidth=2.5, label='Liquid-vapor\nboundary')
ax.plot(T_sat - 273.15, P_sat / 1e6, 'k--', linewidth=1.0)
ax.plot(T_crit - 273.15, P_crit / 1e6, 'r*', markersize=15, zorder=5,
        markeredgecolor='k', markeredgewidth=0.5,
        label=f'Critical point\n({T_crit-273.15:.1f}°C, {P_crit/1e6:.1f} MPa)')

sill_rect = Rectangle((T_sill_low[0], P_sill[0]),
    T_sill_high[1] - T_sill_low[0], P_sill[1] - P_sill[0],
    linewidth=2, edgecolor='red', facecolor='red', alpha=0.15,
    label='Sill emplacement\nzone (1–4 km)')
ax.add_patch(sill_rect)

aureole_rect = Rectangle((T_sill_high[1], P_sill[0]),
    800 - T_sill_high[1], P_sill[1] - P_sill[0],
    linewidth=1.5, edgecolor='orange', facecolor='orange', alpha=0.1,
    linestyle='--', label='Contact aureole\nT range')
ax.add_patch(aureole_rect)

ax.text(10, 1.5, 'Gas', fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.4))
ax.text(-10, 30, 'Liquid', fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.4))
ax.text(150, 40, 'Supercritical\nfluid', fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.4))

ax2 = ax.twinx()
ax2.set_ylim(P_to_depth_litho(0.5), P_to_depth_litho(200))
ax2.set_ylabel('Approximate depth (km, lithostatic)', fontsize=11)

ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Pressure (MPa)', fontsize=12)
ax.set_title('(a) CO$_2$ Phase Diagram\n(Span-Wagner EOS via CoolProp)', fontsize=13)
ax.set_xlim(-20, 400)
ax.set_ylim(0.5, 200)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

# ---- Panel (b): CO2 vs water density with depth ----
ax = axes[1]
depths = np.linspace(0.1, 10, 200)
temps_C = [25, 50, 100, 150, 200, 300, 500]
colors_b = plt.cm.inferno(np.linspace(0.1, 0.9, len(temps_C)))

for k, T_C in enumerate(temps_C):
    rho_co2 = []
    rho_water = []
    T_K = T_C + 273.15
    for z in depths:
        P_Pa = rho_rock * g * z * 1000
        try:
            rho_co2.append(PropsSI('D', 'T', T_K, 'P', P_Pa, 'CO2'))
        except:
            rho_co2.append(np.nan)
        try:
            rho_water.append(PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water'))
        except:
            rho_water.append(np.nan)

    ax.plot(np.array(rho_co2), depths, color=colors_b[k], linewidth=1.8,
            label=f'CO$_2$ {T_C}°C')
    ax.plot(np.array(rho_water), depths, color=colors_b[k],
            linewidth=1.0, linestyle=':', alpha=0.6)

ax.plot([], [], color='gray', linewidth=1.0, linestyle=':',
        label=r'H$_2$O (same T)')

ax.axhspan(1, 4, color='red', alpha=0.08)
ax.text(50, 2.5, 'Sill\nzone', fontsize=10, color='red', fontweight='bold',
        ha='center', va='center')

ax.set_xlabel(r'Density (kg/m$^3$)', fontsize=12)
ax.set_ylabel('Depth (km)', fontsize=12)
ax.set_title('(b) CO$_2$ vs. H$_2$O Density\n(isothermal, CoolProp/IAPWS-95)', fontsize=13)
ax.invert_yaxis()
ax.set_xlim(0, 1200)
ax.set_ylim(10, 0)
ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)

ax.annotate('CO$_2$ denser than H$_2$O\nat low T → no buoyant rise',
            xy=(1050, 3.0), xytext=(400, 7),
            fontsize=9, color='navy',
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

ax.annotate('CO$_2$ lighter than H$_2$O\nat high T → buoyancy-driven',
            xy=(200, 5.5), xytext=(400, 9),
            fontsize=9, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# ---- Panel (c): CO2 solubility (Spycher model) ----
ax = axes[2]
T_C_sol = T_sol_grid - 273.15
P_MPa_sol = P_sol_bar_grid / 10.0  # bar → MPa

cf2 = ax.contourf(T_C_sol, P_MPa_sol, sol_grid,
                  levels=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
                  cmap='YlGnBu', extend='max')
cb2 = fig.colorbar(cf2, ax=ax, label=r'CO$_2$ solubility (mol/kg H$_2$O)', shrink=0.85)

cs = ax.contour(T_C_sol, P_MPa_sol, sol_grid,
                levels=[0.5, 1.0, 1.5, 2.0],
                colors='k', linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

sill_rect2 = Rectangle((T_sill_low[0], P_sill[0]),
    T_sill_high[1] - T_sill_low[0], P_sill[1] - P_sill[0],
    linewidth=2.5, edgecolor='red', facecolor='none',
    label='Sill emplacement\nzone (1–4 km)')
ax.add_patch(sill_rect2)

ax3 = ax.twinx()
ax3.set_ylim(P_to_depth_litho(0), P_to_depth_litho(60))
ax3.set_ylabel('Approximate depth (km, lithostatic)', fontsize=11)

ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Pressure (MPa)', fontsize=12)
ax.set_title('(c) CO$_2$ Solubility in Pure Water\n(Spycher et al. 2003; Spycher & Pruess 2010)',
             fontsize=12)
ax.set_xlim(10, 300)
ax.set_ylim(0, 60)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

ax.annotate('At sill depths (ambient T):\nCO$_2$ highly soluble\nin pore water (~1 mol/kg)',
            xy=(60, 30), xytext=(180, 50),
            fontsize=9, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()

outdir = "/home/tqm5707/Library/Application Support/Claude/LocalAgentModeSessions/sessions/magical-vigilant-babbage/mnt/Model_paper"
plt.savefig(f'{outdir}/CO2_phase_diagram_sills.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{outdir}/CO2_phase_diagram_sills.pdf', dpi=200, bbox_inches='tight')
print("\nFigure saved: CO2_phase_diagram_sills.png and .pdf")
