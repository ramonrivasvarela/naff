# naff.py

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def naff(f: Sequence[float] | np.ndarray, dt: float, nk: int) -> Dict[str, Any]:
    """
    Numerical Analysis of Fundamental Frequencies (NAFF) for a real-valued signal.

    Parameters
    ----------
    f : sequence of float
        Signal samples taken at regular time interval `dt`.
    dt : float
        Sampling interval (time between consecutive samples).
    nk : int
        Number of frequencies to test in the local refinement around the
        dominant FFT frequency. When the dominant frequency is non-zero,
        the algorithm scans [-nk, ..., 0, ..., +nk] offsets. When it is at
        zero, it scans [0, ..., +nk].

    Returns
    -------
    dict
        A dictionary with:
        - "f_final": complex numpy array of the reconstructed single-frequency signal
        - "amplitude": complex amplitude of the fundamental
        - "mu": angular frequency (radians per time unit)
        - "max_index": index (0-based) of the dominant FFT bin
        - "sigmaNu": estimated uncertainty on the frequency
    """
    # Convert input to numpy array
    f_arr = np.asarray(f, dtype=float)
    N = f_arr.size
    if N == 0:
        raise ValueError("Input signal f must be non-empty.")

    # Total duration
    T = dt * N

    # Time series centered around 0, matching the R construction
    half = N / 2.0
    # R: t_series <- ((-(N_2 - 1):N_2) - 0.5) * dT
    # Equivalent:
    t_series = (np.arange(N) - (half - 0.5)) * dt  # shape (N,)

    # Cosine window (same as in the R loop)
    # R: (1 + cos(pi * (i - N_2) / N)), i = 1..N
    i_range = np.arange(1, N + 1)
    window = 1.0 + np.cos(np.pi * (i_range - half) / N)
    f_win = f_arr * window

    # FFT of the windowed signal
    fft_vals = np.fft.fft(f_win)

    # Index of the maximum real part (0-based, unlike R)
    max_index = int(np.argmax(fft_vals.real))

    # Reference angular frequency from FFT bin
    # R: mu_ref <- 2 * (max_index-1) * pi / T
    # Here max_index is already 0-based, so:
    mu_ref = 2.0 * np.pi * max_index / T

    # Frequency step for local refinement
    factor = 2.0 * np.pi / ((nk + 1) * T)

    # Frequency uncertainty estimate (copied from R)
    sigma_nu = 1.0 / (2.0 * T * np.pi * np.sqrt(12.0))

    # Offsets around the reference frequency, in units of `factor`
    if max_index == 0:
        # Peak at zero frequency: scan 0..nk
        offsets = np.arange(0, nk + 1, dtype=float)
    else:
        # Peak away from zero: scan symmetric -nk..nk
        offsets = np.arange(-nk, nk + 1, dtype=float)

    # Candidate angular frequencies
    candidate_mus = mu_ref + offsets * factor  # shape (M,)

    # Compute amplitude at each candidate frequency using the windowed signal
    # For each mu: a(mu) = sum_k f_win[k] * exp(-1j * mu * t_series[k])
    phase = np.outer(candidate_mus, t_series)  # shape (M, N)
    ak_all = np.sum(f_win * np.exp(-1j * phase), axis=1)  # shape (M,)

    # Pick the candidate with maximum magnitude
    idx_best = int(np.argmax(np.abs(ak_all)))
    best_mu = float(candidate_mus[idx_best])

    # Recompute amplitude using the original (unwindowed) signal,
    # as in the R code:
    # amax <- sum_k f[k] * exp(-1i * best_mu * t_series[k]) / N
    phase_best = np.exp(-1j * best_mu * t_series)
    amax = np.sum(f_arr * phase_best) / N

    # Reconstructed single-frequency signal at the best mu
    f_final = amax * np.exp(1j * best_mu * t_series)

    return {
        "f_final": f_final,
        "amplitude": amax,
        "mu": best_mu,
        "max_index": max_index,
        "sigmaNu": sigma_nu,
    }
