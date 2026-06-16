---
title: "HAMON: Passive Optical Sequence Mixing for Long-Horizon Forecasting"
tags: [hardware acceleration, transformer, attention, neural network]
year: 2026
tldr: "HAMON is a passive diffractive optical forecasting core for time-series. Historical values are encoded onto an optical aperture, cascaded trainable phase masks with free-space diffraction shape the forecast directly in the output field. Inference requires a single passive optical propagation pass with no trainable digital sequence-mixing layer. Outperforms digital baselines on ETTm2 and ETTh2 benchmarks (up to 14% MSE improvement)."
wikilinks: [[hardware acceleration]] [[attention]] [[transformer]] [[flash attention]]
---
# HAMON: Passive Optical Sequence Mixing

**Authors**: arXiv:2606.17028
**Year**: 2026

HAMON explores whether the core forecasting operator can be implemented optically rather than as learned digital temporal mixing.

## Method

- Historical values encoded onto an optical aperture
- Future positions left dark
- Cascaded **trainable phase masks** with free-space diffraction shape the forecast in the output field
- Inference: single passive optical propagation pass
- No trainable digital sequence-mixing layer

## Results

- Outperforms strongest digital baselines on **ETTm2** at all horizons
- Outperforms on **ETTh2** at all but the longest horizon
- MSE improvement up to **14%**
- Competitive on Weather benchmark
- Defines a concrete target for passive physical sequence mixing hardware
