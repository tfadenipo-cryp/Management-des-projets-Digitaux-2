
# Car Insurance Analysis Application


## Description
This project is part of the Management of Digital Projects 2 course.
It consists of creating an interactive Python application using Streamlit


to explore and analyze a car insurance dataset.

## Features

- Dataset: Motor Vehicle Insurance Data
- Features: interactive visualizations, filtering, basic statistical analysis
- Deployment: automated CI/CD with Docker

## Variable Definitions

...

## Installation and Usage

### Method 1 : With Docker

1. **Install Docker Desktop**  
   Download and install from [www.docker.com](https://www.docker.com) and make sure Docker is running.

2. **Clone the repository**  
   ```bash
   git clone https://gitlab-mi.univ-reims.fr/fade0003/management-des-projets-digitaux-2.git

    ```
3. **Navigate to the project directory**
    ```bash
    cd management-des-projets-digitaux-2
    ```


4. **Build and run the app**
    ```bash
    docker-compose -f docker-compose.yaml up -d --build
    ```
5. **Launch the app inside the container**
    ```bash
    docker exec -it <container_name> uv run streamlit run main.py

    ```
#### Replace <container_name> with the actual container name (check with docker ps).


### Method 2 : Using Python and uv

1. **Install Python 3.13**
    Download and install from [Python.org](https://www.python.org/)

2. **Install uv**
    Follow instructions at uv docs : https://docs.astral.sh/ub

3. **Clone the repository**
    ```bash
    git clone https://gitlab-mi.univ-reims.fr/fade0003/management-des-projets-digitaux-2.git
    ```

4. **Navigate to the project directory**
    ```bash
    cd management-des-projets-digitaux-2    
    ```


5. **Run the app**

    ```bash
    uv run streamlit run main.py
    ```



## Project Structure

```text
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
├── .gitignore
├── .gitlab-ci.yml
├── .python-version
├── Dockerfile
├── README.Docker.md
├── README.md
├── compose.yaml
├── pyproject.toml
├── requirements.txt
└── uv.lock

```

## CI/CD
The project uses GitLab CI/CD and Docker to automate:
- Dependency installation
- Unit tests execution
- Docker container build and deployment
- Requires 'uv' for pipeline jobs
The pipeline is defined in .gitlab-ci.yml


## Team Members

- Timothe Fadenipo: Data Owner (Owner)
- Matthis Arvois: Data Engineer (Maintainer)
- Nikita Pomozov: Data Gouvernance (Developer)
- Rezi Sabashvilli: Data Scientist (Developer)
- Idriss Jordan: Interface Designer (Developer)
- Cherfatou KOUDOU KIMBA: Data Analyst (Developer)

