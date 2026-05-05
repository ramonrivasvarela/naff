# naff_decomp.py

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np


def naff_single(
    signal: Sequence[float] | np.ndarray,
    dt: float,
    n_refine: int,
) -> Dict[str, Any]:
    """
    One NAFF step: find the dominant frequency of a (possibly complex) signal.

    Parameters
    ----------
    signal : array-like
        Signal samples (real or complex), taken at regular interval dt.
    dt : float
        Sampling interval.
    n_refine : int
        Number of frequencies to test around the dominant FFT bin.

    Returns
    -------
    dict
        {
            "f_final": complex numpy array of the best-fit single-frequency signal,
            "amplitude": complex amplitude A,
            "omega": angular frequency (rad / time),
            "max_index": index (0-based) of the dominant FFT bin,
            "sigmaNu": estimated frequency uncertainty
        }
    """
    # Convert to complex array (handles both real and complex input)
    signal_arr = np.asarray(signal, dtype=np.complex128)
    N = signal_arr.size
    if N == 0:
        raise ValueError("Input signal must be non-empty.")
    if dt <= 0:
        raise ValueError("Sampling interval dt must be positive.")
    if n_refine < 0:
        raise ValueError("n_refine must be non-negative.")

    T = dt * N

    # Time series centered around 0
    # t_series <- ((-(N_2 - 1):N_2) - 0.5) * dT
    half = N / 2.0
    t_series = (np.arange(N) + 0.5) * dt  # shape (N,)

    # Cosine window: (1 + cos(pi * (i - N_2) / N)), i = 1..N
    i_range = np.arange(1, N + 1)
    window = 1.0 + np.cos(np.pi * (i_range - half) / N)
    # Use windowed signal for frequency search
    signal_win = signal_arr * window

    # FFT on windowed signal
    fft_vals = np.fft.fft(signal_win)

    # Index (0-based) of maximum real part
    max_index = int(np.argmax(fft_vals.real))
    if max_index > half:
        max_index = N - max_index

    # Reference angular frequency from FFT bin
    # R: omega_ref <- 2 * (max_index-1) * pi / T  (1-based)
    # Here: omega_k = 2*pi*k/T with k = 0..N-1
    omega_ref = 2.0 * np.pi * max_index / T

    # Frequency step for local refinement
    omega_step = 2.0 * np.pi / ((n_refine + 1) * T) if n_refine > 0 else 0.0

    # If n_refine == 0, we just use the FFT bin directly
    if n_refine == 0:
        best_omega = omega_ref
    else:
        # Offsets around reference bin
        if max_index == 0:
            # Peak at zero frequency: scan [0..n_refine]
            offsets = np.arange(0, n_refine + 1, dtype=float)
        else:
            # Peak away from zero: scan [-n_refine..n_refine]
            offsets = np.arange(-n_refine, n_refine + 1, dtype=float)

        candidate_omegas = omega_ref + offsets * omega_step  # shape (M,)

        # Amplitude for each candidate on the windowed signal:
        # a(omega) = sum_k signal_win[k] * exp(-1j * omega * t_k)
        phase = np.outer(candidate_omegas, t_series)  # (M, N)
        amp_all = np.sum(signal_win * np.exp(-1j * phase), axis=1)

        idx_best = int(np.argmax(np.abs(amp_all)))
        best_omega = float(candidate_omegas[idx_best])

    # Final amplitude using original (un-windowed) signal:
    # amp = sum_k signal[k] * exp(-1j * best_omega * t_k) / N
    phase_best = np.exp(-1j * best_omega * t_series)
    amp = np.sum(signal_arr * phase_best) / N

    return {
        "amplitude": amp,
        "omega": best_omega,
        "max_index": max_index,
    }


def naff_decompose(
    signal: Sequence[float] | np.ndarray,
    dt: float,
    n_components: int,
    n_refine: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iteratively apply NAFF to peel off n_components frequencies.

    Each iteration:
    - Runs NAFF on the current residual.
    - Accumulates the reconstructed component.
    - Stores frequency and amplitude.

    Parameters
    ----------
    signal : array-like
        Real-valued signal samples (will be treated as complex internally).
    dt : float
        Sampling interval.
    n_components : int
        Number of frequency components to extract.
    n_refine : int, optional
        Number of refinement frequencies to test around the dominant FFT peak
        in each NAFF step.

    Returns
    -------
    freqs : np.ndarray, shape (K,)
        Frequencies in Hz (omega / (2*pi)) for K found components
        (K <= n_components).
    amps : np.ndarray, shape (K,)
        Complex amplitudes corresponding to each frequency.
    """
    signal_arr = np.asarray(signal, dtype=np.complex128)
    N = signal_arr.size
    if N == 0:
        raise ValueError("Input signal must be non-empty.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    # Demean (use real part for mean)
    avg = signal_arr.real.mean()
    signal_detr = signal_arr - avg

    # This will accumulate reconstructed components
    recon = np.zeros_like(signal_arr, dtype=np.complex128)
    residual = signal_detr.copy()

    freqs: list[float] = []
    amps: list[complex] = []

    for _ in range(n_components):
        # Run one NAFF step on the current residual
        res = naff_single(residual, dt, n_refine)
        amp = res["amplitude"]
        omega = res["omega"]
        # Update reconstructed signal and residual using the raw NAFF result
        t_series = (np.arange(N) + 0.5) * dt  # shape (N,)
        recon += 2*np.real(amp * np.exp(1j * omega * t_series))
        residual = signal_detr - recon

        freq = omega / (2.0 * np.pi)  # convert to Hz-like units
        freqs.append(freq)
        amps.append(2*amp)
        
    return np.array(freqs, dtype=float), np.array(amps, dtype=np.complex128)
