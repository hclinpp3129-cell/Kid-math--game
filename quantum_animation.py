"""Matplotlib animations of quantum mechanics (saves GIFs).

  python3 quantum_animation.py [output_dir]

1. tunneling.gif   - Gaussian wave packet hitting a barrier it cannot classically cross
2. well_beating.gif - superposition of n=1 and n=2 states in an infinite well
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HBAR = 1.0
M = 1.0
BG, FG, ACCENT, ACCENT2 = "#10151f", "#e6e9ef", "#4cc9f0", "#f72585"


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#445")
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def tunneling(out, dt=0.005, per_frame=200, frames=70, V0=1.5, width=1.0):
    N, L = 2048, 200.0
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    k0 = 1.0
    psi = np.exp(-((x + 40) ** 2) / (2 * 4.0**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    V = np.where(np.abs(x) < width / 2, V0, 0.0)
    half_V = np.exp(-0.5j * V * dt / HBAR)
    kin = np.exp(-1j * HBAR * k**2 * dt / (2 * M))

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=70, facecolor=BG, constrained_layout=True)
    style(ax)
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 0.12)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.fill_between(x, 0, V / V0 * 0.12, color=ACCENT2, alpha=0.5, step="mid", label="barrier")
    (line,) = ax.plot(x, np.abs(psi) ** 2, color=ACCENT, lw=2, label=r"$|\psi|^2$")
    fill = [ax.fill_between(x, 0, np.abs(psi) ** 2, color=ACCENT, alpha=0.25)]
    title = ax.set_title("")
    ax.legend(facecolor=BG, edgecolor="#445", labelcolor=FG, loc="upper right")
    state = {"psi": psi}

    def update(i):
        p = state["psi"]
        for _ in range(per_frame if i else 0):
            p = half_V * np.fft.ifft(kin * np.fft.fft(half_V * p))
        state["psi"] = p
        prob = np.abs(p) ** 2
        line.set_ydata(prob)
        fill[0].remove()
        fill[0] = ax.fill_between(x, 0, prob, color=ACCENT, alpha=0.25)
        T = np.sum(prob[x > width / 2]) * dx
        title.set_text(f"Quantum tunneling   t={i*per_frame*dt:5.1f}   transmitted={T:.1%}")
        return line, title

    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)
    anim.save(out, writer=PillowWriter(fps=15), savefig_kwargs={"facecolor": BG})
    plt.close(fig)


def well_beating(out, frames=60, L=1.0):
    x = np.linspace(0, L, 400)
    phi = lambda n: np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
    E = lambda n: (n * np.pi * HBAR / L) ** 2 / (2 * M)
    period = 2 * np.pi * HBAR / (E(2) - E(1))

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=70, facecolor=BG, constrained_layout=True)
    style(ax)
    ax.set_xlim(0, L)
    ax.set_ylim(-2.2, 3.6)
    ax.set_xlabel("x")
    ax.axhline(0, color="#445", lw=0.8)
    (re_line,) = ax.plot([], [], color=ACCENT2, lw=1.2, alpha=0.8, label=r"Re $\psi$")
    (pr_line,) = ax.plot([], [], color=ACCENT, lw=2.2, label=r"$|\psi|^2$")
    title = ax.set_title("")
    ax.legend(facecolor=BG, edgecolor="#445", labelcolor=FG, loc="upper right")

    def update(i):
        t = period * i / frames
        psi = (phi(1) * np.exp(-1j * E(1) * t / HBAR) + phi(2) * np.exp(-1j * E(2) * t / HBAR)) / np.sqrt(2)
        re_line.set_data(x, psi.real)
        pr_line.set_data(x, np.abs(psi) ** 2)
        title.set_text(r"Infinite well: $(\phi_1+\phi_2)/\sqrt{2}$  probability sloshes back and forth")
        return re_line, pr_line, title

    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)
    anim.save(out, writer=PillowWriter(fps=20), savefig_kwargs={"facecolor": BG})
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    tunneling(out_dir / "tunneling.gif")
    well_beating(out_dir / "well_beating.gif")
    print("saved:", *(p.name for p in sorted(out_dir.glob("*.gif"))))
