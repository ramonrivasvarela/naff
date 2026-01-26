# NAFF Decomposition in Python

This repository implements a small, self-contained Python version of the **Numerical Analysis of Fundamental Frequencies (NAFF)** algorithm and a simple **iterative frequency decomposition** based on it.

The core functions are:

- `naff_single(...)` – one NAFF step: estimate the dominant frequency of a signal.
- `naff_decompose(...)` – repeatedly apply NAFF to extract several frequency components from a real-valued time series.

The implementation follows the original formulation of NAFF introduced by Jacques Laskar.

> J. Laskar, “Frequency analysis for multi-dimensional systems. Global dynamics and diffusion.”  
> *Celestial Mechanics and Dynamical Astronomy* 56, 191–196 (1993).

---

## Features

- Works on real or complex 1D time series sampled at constant time step `dt`.
- Uses a **cosine window** and FFT to find the dominant frequency bin.
- Refines the dominant frequency by scanning a small band around the FFT peak.
- Returns **complex amplitudes** and frequencies in Hz.
- Provides a basic **iterative decomposition**: repeatedly removes the strongest component from the residual.

This is intended as a **lightweight, educational implementation** rather than a fully-optimized production library.

---

## Installation

Clone the repo and make sure you have NumPy installed:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

pip install numpy
