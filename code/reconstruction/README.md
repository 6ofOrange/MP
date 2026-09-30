# MP-LSTM reconstruction code

This folder contains the publication-facing implementation of the model component
used in the manuscript.

Files:

- `mp_lstm_model.py` — four-layer multi-task LSTM architecture with six pollutant-specific heads.
- `train_mp_lstm.py` — data-agnostic training logic, min-max scaling, weighted multi-task MSE, and optional `torchrun`/DDP support.
- `reconstruct_hourly.py` — data-agnostic hourly inference from preprocessed 24-variable predictor grids.

The code intentionally excludes project-specific filesystem paths, raw-data
download logic, geographic constants, and dataset filenames. Predictor assembly,
quality control, temporal harmonization, and the derivation of relative humidity
are described in the manuscript and Supplementary Methods.

Model settings represented here follow the manuscript: 24 predictors, 48-hour
sequences, four LSTM layers, hidden size 256, dropout 0.01, six pollutant outputs,
Adam learning rate 2e-4, batch size 256, 18 epochs, and an SO2 task weight of 0.8.