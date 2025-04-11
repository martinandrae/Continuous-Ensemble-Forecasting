# Code for the ICLR 2025 paper [Continuous Ensemble Weather Forecasting with Diffusion models](https://arxiv.org/abs/2410.05431)

This repository contains all code needed to train and evaluate the models proposed in the paper.

## Prerequisites

The code should be runnable with a standard setup of pytorch with some additional packages listed below.

- pytorch
- pandas
- numpy
- tqdm
- matplotlib
- zarr
- xarray
- jupyter
- ipykernel
- cartopy (for plotting)
- ffmpeg (for animations)

## Data Preparation

First, download ERA5 data with 5.625deg from [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895). The data directory should look like the following
```
era5_data
   |-- 10m_u_component_of_wind
   |-- 10m_v_component_of_wind
   |-- 2m_temperature
   |-- constants
   |-- geopotential_500
   |-- temperature_850
```

The data is loaded as ``.npy`` files instead of ``netcdf`` so you need to run the ``create_dataset.py`` script.


## Training

To train a model, locate a relevant ``config.json`` file under ``configs/train`` and run
```
python train.py configs/train/config.json
```

See the ``guide.json`` for explanations of all config parameters.

Some of the models trained for the paper are available as checkpoints under ``models/``.

## Generation and Evaluation

To generate and save predictions from a trained model, locate a relevant ``config.json`` file under ``configs/predict`` and run
```
python predict.py configs/predict/config.json
```

See the ``guide.json`` for explanations of all config parameters.

This also evaluates the model and saves metrics to a ``zarr` file.

## Plotting

The notebook ``plot.ipynb`` contains code to visualize metrics and forecasts.

## If you use this code for some purpose, please cite:

```
@inproceedings{
    andrae2025continuous,
    title={Continuous Ensemble Weather Forecasting with Diffusion models},
    author={Martin Andrae and Tomas Landelius and Joel Oskarsson and Fredrik Lindsten},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=ePEZvQNFDW}
}
```