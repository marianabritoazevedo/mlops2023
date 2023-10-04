# Project 2 - Airflow Data Pipeline to Download Podcasts 🎧

## ℹ️ About this Project

The Project 02 involves building a data pipeline using the Apache Airflow tool for downloading podcasts. This portfolio project is available on the [Dataquest.io](https://app.dataquest.io/) platform. It is designed to teach how to create a data pipeline using Airflow, construct a DAG (Directed Acyclic Graph), and define tasks to perform various activities within this pipeline. By the completion of this project, you will be able to visualize the following pipeline in Apache Airflow:

![img](./img/pipeline-airflow.png)

## 🚀 How to Run this Project

There are many steps to install the required libraries and run Apache Airflow. To make it easy, it's recommended that you do the following tutorial in at Linux Operating System.

### 1️⃣ Running Apache Airflow

1. **Installing the libraries**

First, install the `sqlite3` library to use as the database for this project:

```
sudo apt install sqlite3
```

Afterward, create a virtual environment using the `requirements.txt` file in this directory. Activate the virtual environment and install the necessary libraries:

```
virtualenv my_env

source my_env/bin/activate

pip install -r requirements.txt
```

2. **Installing Airflow**

First, define the airflow home path:

```
export AIRFLOW_HOME=~/airflow
```

Specify the versions of Airflow and Python. In this tutorial, version `2.3.1` of Airflow is used:

```
AIRFLOW_VERSION=2.3.1
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
```

Create the URL for installation, considering the specified Airflow and Python versions:


```
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

Finally, install Apache Airflow:

```
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

3. **Running Airflow**

After installing all the required libraries, run a standalone version of Apache Airflow with the following command:

```
airflow standalone
```

Your terminal will display a detailed log, including your credentials (usually with the username `admin`) to access Airflow. Note down these credentials, access the URL `localhost:8080`, and log in. You should be able to view the Apache Airflow dashboard after successful login. 

### 2️⃣ Necessary configurations in the project

### 3️⃣ Running the project