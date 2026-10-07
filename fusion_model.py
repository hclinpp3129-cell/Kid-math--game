"""核融合模型 (D-T fusion model).

  python3 fusion_model.py [output_dir]

Outputs
  fusion_cross_section.png - D-T / D-D / D-He3 cross-sections (Bosch-Hale-like fits) vs. Gamow tunneling
  fusion_reactivity.png    - Maxwellian <sigma v>(T) and the Lawson triple product n*T*tau_E needed for ignition
  fusion_plasma.gif        - 2D ions in a magnetic-confinement-like trap; hot D-T ions that collide fuse
                             into He-4 + neutron (14.1 MeV), and the energy is shown as a running tally

Physics
  Coulomb barrier  : U = Z1 Z2 e^2 / (4 pi eps0 r)  ~ 0.4 MeV for D-T, yet fusion happens at ~10 keV
                     because of quantum tunneling (Gamow factor exp(-sqrt(E_G/E))).
  Cross-section    : sigma(E) = S(E)/E * exp(-sqrt(E_G/E)),  E_G = 2 mu c^2 (pi alpha Z1 Z2)^2
  Reactivity       : <sigma v> = sqrt(8/(pi mu)) (kT)^-3/2 * Int sigma(E) E exp(-E/kT) dE
  Lawson (ignition): n T tau_E >= 12 T^2 / (E_alpha <sigma v>)   [E_alpha = 3.5 MeV charged-product energy]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ALPHA_FS = 1 / 137.036
AMU_KEV = 931494.0  # keV per atomic mass unit
KEV = 1.602176634e-16  # J
BARN = 1e-28  # m^2
BG, FG, ACCENT, ACCENT2, ACCENT3 = "#10151f", "#e6e9ef", "#4cc9f0", "#f72585", "#ffd166"

# reaction: (Z1, Z2, m1[u], m2[u], S0 [keV*barn], energy release [MeV], charged-product energy [MeV])
REACTIONS = {
    "D-T":   (1, 1, 2.0141, 3.0160, 1.2e4, 17.6, 3.5),
    "D-D":   (1, 1, 2.0141, 2.0141, 56.0, 3.65, 3.65),  # avg of the two branches
    "D-He3": (1, 2, 2.0141, 3.0160, 6.0e3, 18.3, 18.3),
}


def reduced_mass_keV(m1, m2):
    return m1 * m2 / (m1 + m2) * AMU_KEV


def gamow_energy(z1, z2, mu_keV):
    """Gamow energy E_G in keV."""
    return 2 * mu_keV * (np.pi * ALPHA_FS * z1 * z2) ** 2


def cross_section(E_keV, name):
    """sigma(E) in barns. D-T has the well-known 64 keV resonance (modelled as a Breit-Wigner bump)."""
    z1, z2, m1, m2, S0, _, _ = REACTIONS[name]
    EG = gamow_energy(z1, z2, reduced_mass_keV(m1, m2))
    S = S0 * np.ones_like(E_keV)
    if name == "D-T":
        S = 1.2e2 + 5.2e4 * 1.0 / (1 + ((E_keV - 64.0) / 25.0) ** 2) * 1.0
    return S / E_keV * np.exp(-np.sqrt(EG / E_keV))


def reactivity(T_keV, name):
    """Maxwellian-averaged <sigma v> in m^3/s (numerical integration)."""
    z1, z2, m1, m2, *_ = REACTIONS[name]
    mu = reduced_mass_keV(m1, m2)
    c = 2.998e8
    out = []
    for T in np.atleast_1d(T_keV):
        E = np.linspace(0.05, 40 * T, 6000)
        integrand = cross_section(E, name) * BARN * E * np.exp(-E / T)
        integral = np.trapezoid(integrand, E)  # barn*keV^2 -> m^2 keV^2
        pref = np.sqrt(8 / (np.pi * mu)) * c * T ** -1.5  # (keV^-1/2)(m/s) keV^-3/2
        out.append(pref * integral)
    return np.array(out)


def lawson_triple(T_keV, name="D-T"):
    """Ignition triple product n*T*tau_E in keV s / m^3."""
    ealpha = REACTIONS[name][6] * 1e3  # keV
    sv = reactivity(T_keV, name)
    return 12 * T_keV**2 / (ealpha * sv)


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#445")
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def plot_cross_section(out):
    E = np.logspace(0, 3, 400)  # 1 keV .. 1 MeV
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110, facecolor=BG, constrained_layout=True)
    style(ax)
    for (name, col) in zip(REACTIONS, (ACCENT2, ACCENT, ACCENT3)):
        ax.loglog(E, cross_section(E, name), color=col, lw=2, label=name)
    ax.axvline(400, color="#889", ls=":", lw=1)
    ax.text(420, 3e-9, "classical Coulomb\nbarrier ≈ 0.4 MeV", color=FG, fontsize=8)
    ax.set_ylim(1e-10, 10)
    ax.set_xlabel("center-of-mass energy E (keV)")
    ax.set_ylabel(r"cross-section $\sigma$ (barn)")
    ax.set_title("Fusion cross-section: tunneling beats the Coulomb barrier")
    ax.legend(facecolor=BG, edgecolor="#445", labelcolor=FG)
    ax.grid(alpha=0.2, which="both")
    fig.savefig(out / "fusion_cross_section.png", facecolor=BG)
    plt.close(fig)


def plot_reactivity(out):
    T = np.logspace(0, 2.3, 60)  # 1..200 keV
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=110, facecolor=BG, constrained_layout=True)
    for a in (a1, a2):
        style(a)
        a.grid(alpha=0.2, which="both")
    for (name, col) in zip(REACTIONS, (ACCENT2, ACCENT, ACCENT3)):
        a1.loglog(T, reactivity(T, name), color=col, lw=2, label=name)
    a1.set_xlabel("ion temperature T (keV)")
    a1.set_ylabel(r"$\langle\sigma v\rangle$ (m$^3$/s)")
    a1.set_title("Maxwellian reactivity")
    a1.set_ylim(1e-30, 1e-21)
    a1.legend(facecolor=BG, edgecolor="#445", labelcolor=FG)

    Tl = np.linspace(3, 100, 300)
    tri = lawson_triple(Tl)
    a2.semilogy(Tl, tri, color=ACCENT2, lw=2)
    i = np.argmin(tri)
    a2.plot(Tl[i], tri[i], "o", color=ACCENT3)
    a2.annotate(f"min ≈ {tri[i]:.1e} keV·s/m³\nat T ≈ {Tl[i]:.0f} keV", (Tl[i], tri[i]),
                (Tl[i] + 8, tri[i] * 6), color=FG, fontsize=9, arrowprops=dict(arrowstyle="->", color=FG))
    a2.fill_between(Tl, tri, tri.max() * 10, color=ACCENT, alpha=0.12)
    a2.text(55, tri.max() * 3, "ignition", color=ACCENT, fontsize=11)
    a2.set_xlabel("ion temperature T (keV)")
    a2.set_ylabel(r"$n T \tau_E$ needed (keV s / m$^3$)")
    a2.set_title("Lawson triple product (D-T ignition)")
    fig.savefig(out / "fusion_reactivity.png", facecolor=BG)
    plt.close(fig)
    return tri[i], Tl[i]


def plasma_gif(out, n=90, frames=120, seed=3):
    """Toy 2D plasma: D (blue) and T (yellow) ions bounce in a box; close hot pairs fuse stochastically."""
    rng = np.random.default_rng(seed)
    L = 10.0
    pos = rng.uniform(1, L - 1, (n, 2))
    vel = rng.normal(0, 1.0, (n, 2))
    kind = np.array([0] * (n // 2) + [1] * (n - n // 2))  # 0 = D, 1 = T
    alive = np.ones(n, bool)
    fx = []  # fusion flashes: [x, y, age]
    he = []  # He-4 products (static dots) and neutrons (flying out)
    neutrons = []
    energy = {"count": 0}
    dt, r_hit = 0.05, 0.35

    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=70, facecolor=BG, constrained_layout=True)
    style(ax)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])
    sc = ax.scatter([], [], s=30)
    flash = ax.scatter([], [], s=[], c=ACCENT3, alpha=0.8)
    nsc = ax.scatter([], [], s=14, c="#ffffff")
    hsc = ax.scatter([], [], s=24, c=ACCENT2)
    txt = ax.text(0.2, L - 0.5, "", color=FG, fontsize=9, va="top")
    ax.set_title("D-T fusion:  D + T → He-4 (3.5 MeV) + n (14.1 MeV)")

    def step():
        pos[:] += vel * dt
        for d in (0, 1):  # reflect off walls (confinement)
            lo, hi = pos[:, d] < 0.2, pos[:, d] > L - 0.2
            vel[lo | hi, d] *= -1
            pos[lo, d] = 0.2
            pos[hi, d] = L - 0.2
        vel[:] += -0.15 * (pos - L / 2) * dt  # soft magnetic-well pull toward centre
        D = np.where(alive & (kind == 0))[0]
        T = np.where(alive & (kind == 1))[0]
        if len(D) and len(T):
            dist = np.linalg.norm(pos[D][:, None] - pos[T][None], axis=2)
            for a, b in zip(*np.where(dist < r_hit)):
                i, j = D[a], T[b]
                if not (alive[i] and alive[j]):
                    continue
                rel = np.sum((vel[i] - vel[j]) ** 2)
                if rng.random() < min(1.0, 0.04 * rel):  # hotter -> more tunneling
                    alive[i] = alive[j] = False
                    c = (pos[i] + pos[j]) / 2
                    fx.append([c[0], c[1], 0])
                    he.append(c.copy())
                    ang = rng.uniform(0, 2 * np.pi)
                    neutrons.append([c[0], c[1], 4 * np.cos(ang), 4 * np.sin(ang)])
                    energy["count"] += 1
        # plasma heating by alphas: stay hot
        vel[alive] *= 1.0 + 0.0008 * (np.linalg.norm(vel[alive], axis=1) < 1.2)[:, None]

    def update(i):
        if i:
            for _ in range(3):
                step()
        for f in fx:
            f[2] += 1
        for nu in neutrons:
            nu[0] += nu[2] * dt * 3
            nu[1] += nu[3] * dt * 3
        a = alive
        sc.set_offsets(pos[a])
        sc.set_color([ACCENT if k == 0 else ACCENT3 for k in kind[a]])
        live_fx = [f for f in fx if f[2] < 8]
        flash.set_offsets(np.array([[f[0], f[1]] for f in live_fx]) if live_fx else np.empty((0, 2)))
        flash.set_sizes([60 + 40 * f[2] for f in live_fx])
        flash.set_alpha(0.6)
        vis = [nu[:2] for nu in neutrons if 0 < nu[0] < L and 0 < nu[1] < L]
        nsc.set_offsets(np.array(vis) if vis else np.empty((0, 2)))
        hsc.set_offsets(np.array(he) if he else np.empty((0, 2)))
        txt.set_text(f"fusion events: {energy['count']}\n"
                     f"energy released: {17.6 * energy['count']:.1f} MeV\n"
                     "blue = D   yellow = T   pink = He-4   white = n")
        return sc, flash, nsc, hsc, txt

    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)
    anim.save(out / "fusion_plasma.gif", writer=PillowWriter(fps=16))
    plt.close(fig)
    return energy["count"]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fusion_outputs")
    out.mkdir(parents=True, exist_ok=True)

    for name, (z1, z2, m1, m2, *_rest) in REACTIONS.items():
        EG = gamow_energy(z1, z2, reduced_mass_keV(m1, m2))
        print(f"{name:6s} Gamow energy E_G = {EG/1e3:6.2f} MeV, Q = {REACTIONS[name][5]} MeV")

    plot_cross_section(out)
    tri, Tmin = plot_reactivity(out)
    sv10 = reactivity(10.0, "D-T")[0]
    print(f"D-T <sigma v> at 10 keV = {sv10:.2e} m^3/s")
    print(f"Lawson minimum n*T*tau_E = {tri:.2e} keV s/m^3 at T = {Tmin:.0f} keV")
    n = plasma_gif(out)
    print(f"plasma animation: {n} fusion events -> {out}/")


if __name__ == "__main__":
    main()
