# IT3385_ML_Operations

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

IT3385 Assignment 1
dataset: https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset<br>
github: https://github.com/keoner/IT3385_mlops/tree/Keoni<br>
website: https://it3385-mlops-hool.onrender.com/

## Deployment Guide

### Prerequisites

Before running the application, ensure the following are installed:

- Python 3.10
- Poetry
- Git

### 1. Clone the Repository

Clone the project repository and navigate to the project directory.

```bash
git clone <repository-url>
cd it3385_mlops
```

### 2. Install Dependencies

The project's dependencies are defined in `pyproject.toml` and `poetry.lock`.

Install all required dependencies using:

```bash
poetry install
```

### 3. Run the Streamlit Application

From the project root directory, run:

```bash
poetry run streamlit run app/app.py
```

The application will start locally and can typically be accessed at:

```text
http://localhost:8501
```

The trained models are automatically retrieved from the configured Hugging Face repository when required by the application.

### 4. Run DVC

Get the csv file from google drive
```bash
./setup.ps1
```

### 5. Run MLflow

To view experiment tracking results, start the MLflow server from the project root directory:
```bash
./start_mlflow.ps1
```

## User Guide

The Predictive Maintenance Dashboard provides three machine learning functions:

- Maintenance Classification
- Time-to-Failure Regression
- Anomaly Detection

### Data Input

Users can select between two input methods from the sidebar.

#### Manual Input

Select **Manual input** and enter the machine's sensor readings using the provided controls:

- Temperature
- Vibration
- Humidity
- Pressure
- Energy Consumption
- Machine ID
- Timestamp

The entered values will be used to generate a prediction for a single machine observation.

#### CSV Upload

From the data/samples folder, Select **Upload CSV** and upload a `.csv` file containing the required sensor data.

The application will validate the uploaded dataset and display a preview before prediction.

The CSV must contain the following sensor columns:

```text
temperature
vibration
humidity
pressure
energy_consumption
```

## Project Organisational Structure

Team Leader: Keoni
Team Member: Evan

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         it3385_mlops and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── config             <- Configuration folder for hydra
│   └── config.yaml    <- Configuration file for hydra 

```

--------

