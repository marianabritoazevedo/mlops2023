# Project 03 - Predicting Heart Disease with Supervised Machine Learning 🩺❤️🤖

## ℹ️ About this Project

The project 03 is about to build a KNN classifier to predict heart diseases. This guided project, avaiable on the [Dataquest.io](https://app.dataquest.io/) platform, focuses on analyzing a dataset containing various features related to numerous patients, such as cholesterol levels, age, and ECG readings. The project encompasses a comprehensive data pipeline, including exploratory data analysis (EDA) to understand the dataset, data cleaning, feature selection, classifier construction, and result prediction

## 🚀 How to Run this Project

1. **Running `project1.py`**

    - First, create a virtual environment on your computer:

        ```bash
        python -m venv your_env
        ```

    - In the directory `project03`, there is the `requirements.txt` file. Install the required libraries:

        ```bash
        pip install -r requirements.txt
        ```

    - Execute the code by running the following command:

        ```bash
        python project3.py
        ```

    - To assess the code quality using `pylint`, run the following command:

        ```bash
        pylint project3.py
        ```

    - For unit testing, use the following command:

        ```bash
        pytest project3.py
        ```

3. **Running `project3_notebook.ipynb`**

    - Open the Jupyter notebook, `project3_notebook.ipynb`, on your local machine or in a tool like Google Colaboratory.

    - Run all the notebook cells to perform necessary installations and interact with the code.

## ⚠️ Note about pylint

In this project, after running  `pylint`, the obtained score was 9.69 out of 10.00. This occurred due to three main reasons:

- Variables `X` and `y` don't adhere to the snake_case naming style: as these variables are standard in the data science community, the decision was made to retain these names;

- To many local variables in `main()` function: given the complexity of the analysis being performed, and with no repeated variables within this function, it was decided to keep it as is.

Feel free to explore the code and learn the best code practices for this project! 💻