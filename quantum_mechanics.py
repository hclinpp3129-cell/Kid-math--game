"""Quantum mechanics demos (numpy only).

1. Time-dependent Schrodinger equation (split-step Fourier): wave packet tunneling
2. Particle in an infinite well: energy levels via finite-difference Hamiltonian
3. Qubit: superposition, interference, and measurement (Born rule)
"""
import numpy as np

HBAR = 1.0
M = 1.0


# ---------- 1. Wave packet tunneling ----------
def tunneling(steps=14000, dt=0.005, barrier_height=1.5, barrier_width=1.0):
    N, L = 2048, 200.0
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    k0 = 1.0  # mean momentum -> E = k0^2/2 = 0.5 < barrier height
    psi = np.exp(-((x + 40) ** 2) / (2 * 4.0**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

    V = np.where(np.abs(x) < barrier_width / 2, barrier_height, 0.0)
    half_V = np.exp(-0.5j * V * dt / HBAR)
    kinetic = np.exp(-1j * HBAR * k**2 * dt / (2 * M))

    for _ in range(steps):
        psi = half_V * np.fft.ifft(kinetic * np.fft.fft(half_V * psi))

    prob = np.abs(psi) ** 2
    T = np.sum(prob[x > barrier_width / 2]) * dx
    R = np.sum(prob[x < -barrier_width / 2]) * dx
    print(f"[Tunneling] E={0.5*k0**2:.2f} < V0={barrier_height}: "
          f"transmitted={T:.3f}, reflected={R:.3f}, total={np.sum(prob)*dx:.3f}")


# ---------- 2. Infinite square well ----------
def infinite_well(n_levels=5, L=1.0, N=500):
    x = np.linspace(0, L, N + 2)[1:-1]
    dx = x[1] - x[0]
    diag = np.full(N, HBAR**2 / (M * dx**2))
    off = np.full(N - 1, -HBAR**2 / (2 * M * dx**2))
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    E = np.linalg.eigvalsh(H)[:n_levels]
    print("[Infinite well] n  numeric   analytic")
    for n, e in enumerate(E, 1):
        exact = (n * np.pi * HBAR / L) ** 2 / (2 * M)
        print(f"                {n}  {e:8.4f}  {exact:8.4f}")


# ---------- 3. Qubit ----------
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
KET0 = np.array([1, 0], dtype=complex)


def measure(state, shots=10000, rng=np.random.default_rng(0)):
    p = np.abs(state) ** 2
    outcomes = rng.choice([0, 1], size=shots, p=p / p.sum())
    return np.bincount(outcomes, minlength=2) / shots


def qubit():
    plus = H_GATE @ KET0
    print(f"[Qubit] H|0> amplitudes={np.round(plus, 3)}, measured P(0,1)={measure(plus)}")
    back = H_GATE @ plus  # interference: amplitudes for |1> cancel
    print(f"        H H|0> measured P(0,1)={measure(back)}  (interference restores |0>)")


if __name__ == "__main__":
    tunneling()
    infinite_well()
    qubit()
