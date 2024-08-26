# Kaggle Commonlit Readability Prize 

## Summary 
In this project, we aim to **predict the readability measures** of the given texts by 3 different learning algorithmns - Ridge regression, Support vector regression and BERT. :octocat:
For more information on the project background and data access: https://www.kaggle.com/c/commonlitreadabilityprize
---
## Contributors 

- WENG XINN CHOW (z5346077) <z5346077@unsw.edu.au>
- ZIYAN GUAN (z5342540) <z5342540@unsw.edu.au>

## Python files (with usage)
1. Commonlit-prelim.py
    * intput file: raw training data *train.csv*
    * main modules: *spacy*, *textstat*
    * expected output: EDA (stats & plots); Extract basic linguistic features to a csv file

2. Commonlit-Aztertest.py
    * input file: raw training data *train.csv*, the program will split excerpt into separate text files
    * main modules: *selenium*
    * relevant information: https://github.com/kepaxabier/AzterTest
    * expected output: one CSV file consisting of 164 Aztertest linguistic features for all sample excerpts

3. Commonlit-BERT.py
    * input file: raw training data *train.csv*
    * main modules: *transformers*, *tensorflow-gpu*, *sklearn*
    * suggested environment: Google Colab (GPU)
    * reference: https://colab.research.google.com/drive/1cV516YJdolaABHgkBUoI0mMhN08tZkga?usp=sharing
    * expected output: Train, Validation and Test RMSE of BERT models; Output model weight to folder; Output the test prediction of BERT model as test_prediction_bert.csv

4. comp9417_feature_engineering.py
    * input file: *train.csv*, *Aztertest_features.csv*
    * packages installed: *textstat*, *spacy*, *collinearity*
    * main modules: *textstat*, *spacy*, *collinearity*
    * expected output: Create basic linguistic features and combine with the existing features from Aztertest; Run the collinearity and correlation analyses; Save them as train_features.csv

5. comp9417_lr_svr.py
    * input file: *train_features.csv*, *train.csv*
    * main modules: *sklearn*, *statsmodel*
    * expected output: Train, Validation and Test RMSE of Linear Regression and SVR models; Output the statistic summary to find the most significant features; Output the test predictions of Linear Regression and SVR models as test_prediction_ridge.csv and test_prediction_svr.csv

## Submitted files
1. train.csv
Raw original training file provided in the Kaggle competition. We derive features based on the excerpts in this file and also split this file into **training, validation and test** dataset to evaluate our models.

2. Aztertest_features.csv
This feature files includes all 165 features extracted from Aztertest online webtool for all samples in the train.csv. It is the output file of Commonlit-Aztertest.py

3. train_features.csv
This feature file combines Aztertest_features.csv and some additional features, adding up to 176 explicit features in total. It is the output file of comp9417_feature_engineering.py and the input file of linear regression & svr modelling comp9417_lr_svr.py

4. test_prediction_bert.csv
Prediction output of the best BERT model on the test data (we split train.csv to derive the test data)

5. test_prediction_ridge.csv
Prediction output of the best Ridge regression model on the test data (we split train.csv to derive the test data)

6. test_prediction_svr.csv
Prediction output of the best Support vector regression model on the test data (we split train.csv to derive the test data)