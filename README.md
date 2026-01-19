# CURE: Calibrated Uncertainty in Restoration via Spectral Estimation

Analyzing and correcting calibration failures in diffusion-based image restoration.

## Quick Start

```bash
pip install -r requirements.txt
python src/degradations.py  # Test degradations
python -c "from src.wiener import test_wiener; test_wiener()"  # Test Wiener
jupyter notebook notebooks/01_exploration.ipynb
```

## Project Structure

```
src/
├── degradations.py   # Image degradation operators
├── wiener.py         # Classical Wiener filter (calibration reference)
├── dps.py            # Diffusion posterior sampling
├── calibration.py    # Calibration metrics (ECE, coverage)
├── frequency.py      # Frequency band utilities
├── cure.py           # Recalibration model
└── utils.py          # I/O and visualization
```

## Core Hypothesis

Diffusion models are miscalibrated at frequencies where |H(f)| ≈ 0.
