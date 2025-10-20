# Car Insurance Analysis Application



## Description

This project is part of Management of Digital Projects 2 course. It consists of creating an interactive Python application using Streamlit to explore and analyze a car insurance dataset.


Dataset: Motor Vehicle Insurance Data

Features: interactive visualizations, filtering, basic statistical analysis

Deployment: automated CI/CD with Docker

## Installation

### Clone the repository:

git clone https://gitlab-mi.univ-reims.fr/fade0003/management-des-projets-digitaux-2.git
cd management-des-projets-digitaux-2

### Create a python vitrual environment :
python -m venv .venv

### Activate the environment :
Windows :
.venv\Scripts\activate
MacOs/Linux :
source .venv/bin/activate

### Install dependencies : 
pip install -r requirements.txt

## Run the Application 

### With Streamlit :
streamlit run src/main.py

### With Docker :

docker build -t mpd2-app .
docker run -p 8501:8501 mpd2-app

## Project Structure : 

.
├── .venv/                  # Virtual environment
├── data/
│   ├── raw/                # Raw data
│   │   └── Motor_vehicle_insurance_data.csv
│   └── processed/          # Processed data
├── src/
│   ├── main.py             # Streamlit entry point
│   └── functions/
│       ├── __init__.py
│       └── dashboard.py    # Dashboard functions
├── tests/                  # Unit tests
│   └── test_example.py
├── .dockerignore
├── .editorconfig
├── .gitingore
├── .gitlab-ci.yml
├──.python-version
├── Dockerfile
├── README.Docker.md
├── README.md
├── compose.yaml
├── pyproject.toml
├── requirements.txt
├── uv.lock




## CI/CD

The project uses GitLab CI/CD and Docker to automate:

Dependency installation

Unit tests execution

Docker container build and deployment

The pipeline is defined in .gitlab-ci.yml.


## Team Members

Timothe Fadenipo : Owner
Matthis Arvois : Maintainer
Nikita Pomozov : Developer
Rezi Sabashvilli : Developer
Idriss Jordan : Developer 
Cherfatou : Developer
