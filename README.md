# Credit Risk Analysis Report

## Overview & Features

This analysis was completed using various Supervised Machine Learning tools and techniques. More details:

*Purpose*:
* This is analysis was conducted to build a model that can identify the creditworthiness of borrowers.

*Data Information*:
* A dataset of historical lending activity from peer-to-peer lending services company was used for prediction of imbalanced classes.

*Variables to be Predicted*:
* Using tools like ```value_counts()``` and ```train_test_split()``` provided a streamlined approach towards further data prediction and analysis.

---

## Stages of the Machine Learning Process:

### Data Preparation

When beginning our analysis, we sliced the data we needed from the original DataFrame into new DataFrames that would integrate into the machine learning tools provided by SciKit Learn.

![slice](https://github.com/antonmaliksi/FinTechModule12Challenge/blob/main/Readme%20Resources/slice.PNG)

We then use the following code to split the data into the necessary training and testing variables: <br>
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    ```
<br>Doing so allows us to move onto the next step: Logistic Regression.

### Logistic Regression

Using SciKit Learn's ```LogisticRegression()``` function, we use the standard "model-fit-predict" structure to provide the following information:

![OGdata](https://github.com/antonmaliksi/FinTechModule12Challenge/blob/main/Readme%20Resources/OGdata.PNG)

### Resampled Data using OverSampling

When one class of data does not have enough data within its training set, the model will often generate skewed or "biased" predictive data towards the class with more data. To avoid this bias, we use Random Oversampling which randomly selects data instances of the minority class and adds them to the original training set to achieve a 50-50 balance between both classes.

To accomplish this, we use IMBLearn's ```RandomOverSampler()``` function to fit the original data into a resampled data set:

    ```python
    X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)
    ```
    
Using the same "model-fit-predict" structure used in the original prediction but with the resampled variables, we are provided the following:

![REdata](https://github.com/antonmaliksi/FinTechModule12Challenge/blob/main/Readme%20Resources/REdata.PNG)

We can conduct a final visual comparison of the two models of Supervised Machine Learning Data:

Original Data                        |  Resampled Data
:----------------------------------------:|:----------------------------------------:
![OGdata](https://github.com/antonmaliksi/FinTechModule12Challenge/blob/main/Readme%20Resources/OGdata.PNG)  | ![REdata](https://github.com/antonmaliksi/FinTechModule12Challenge/blob/main/Readme%20Resources/REdata.PNG)

## Results

**Machine Learning Model 1 (Original Data)**:
* *Accuracy*: 95%
* *Precision*: 99%
* *Recall*: 99%

**Machine Learning Model 2 (Resampled Data)**:
* *Accuracy*: 99%
* *Precision*: 99%
* *Recall*: 99%

## Summary
When referencing the pure empirical data of both Machine Learning Models, it's evident that Model 2 (Resampled Data) had higher accuracy and overall higher scores provided by the classification report.
While the target of our analysis was to predict both "0" and "1", the resampled data proved that the performance of both classes were equally important to create a more accurate prediction.

---

## Technologies

This notebook utilizes **Python (v 3.9.7)** and the following libraries:

1. pandas
2. numpy
3. Path from pathlib
4. balanced_accuracy_score and confusion_matrix from sklearn.metrics
5. classification_report_imbalanced from imblearn.metrics
6. warnings

---

## Installation Guide
Pandas, Numpy, and Pathlib should be part of the base applications that were installed with the Python version above; if not, you will have to install them through the pip package manager of Python.

To install imbalance-learn, run the following:

   
    conda install -c conda-forge imbalanced-learn
   

To install scikit-learn, run the following:

    
    conda install -c conda-forge scikit-learn
    
    
    
After installing, run the following to ensure that sklearn and hvplot are installed:

    
    conda list imbalanced-learn
    
    conda list scikit-learn
    
    
If any errors occur, please contact IT for further assistance.

---

## User Guide
To use the notebook:

### Load the Data
1. Open "credit_risk_resampling.ipynb"
2. Look for the following code:
    ```python
    loans_df = pd.read_csv(Path("Resources/lending_data.csv"))
    ```
Ensure you have the correct .CSV file imported from within the Resources folder, which can be located in the parent folder.

---

## Versioning History
All Github commits/pulls were conducted and verified by Anton Maliksi.

---

## Contributors
Anton Maliksi was the sole contributor for this notebook.

---

## Licenses
No licenses were used for this project.