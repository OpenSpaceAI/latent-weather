# Latent-Weather

This is the code repository for article **Accurate Medium-to-Long-Term Weather Forecasting in Latent Space**.

## Earth Grid

### Dataset

The Earth Grid dataset is from [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), which can be downloaded via [Google Cloud](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr).

After downloading, you can use `dataset_process/earth_grid_1.py` and `dataset_process/earth_grid_2.py` to convert these raw data to pytorch tensor format and then normalize them, so that the model can use these processed data for training and testing.

### Train

Run `earth_grid_ae/main.py` to train the Spatial-Encoder and Spatial-Decoder, then run `earth_grid_f/main.py` to train the Temporal-Predictor.

## Earth Station

### Dataset

The Earth Station dataset uses pre-processed data provided by [Corrformer](https://codeocean.com/capsule/0341365/tree/v1). Here we directly provide the pytorch tensor data in `dataset/earth_station/tensor` directory since the file size is not too large.

### Train

Run `earth_station_ae/main.py` to train the Spatial-Encoder and Spatial-Decoder, then run `earth_station_f/main.py` to train the Temporal-Predictor.

## Mars Grid

### Dataset

The Mars Grid dataset is from [OpenMARS](https://ordo.open.ac.uk/articles/dataset/OpenMARS_continuous_MY28-35_standard_database/24573205).

Similarly, after downloading, you can use `dataset_process/mars_grid_1.py` and `dataset_process/mars_grid_2.py` to do format conversion and normalization (not provided yet).

### Train

Run `mars_grid_ae/main.py` to train the Spatial-Encoder and Spatial-Decoder, then run `mars_grid_f/main.py` to train the Temporal-Predictor.
