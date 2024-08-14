# +        __Ed Machina__ 

---

#### .1.     **Ed_Machina_RAW_EDA.ipynb**     *(Updated)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16HkTb7CGf8AfLWmzKy9wxFRhyaWFhna_?usp=sharing)


#### .2.     **Ed_Machina_PREPROCESSING_EDA.ipynb**     *(Updated)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x7SkJnYftjmRSBVv0NIRDZJCcHluDnYW?usp=sharing)


#### .3.     **APP DIAGRAM**

```
.
├── .env                                    -> environment variables
├── .gitignore                              -> gitignore file
├── datasets/                               -> datasets folder
│   ├── challenge_edMachina.csv
│   ├── parquet/
│   └── samples/
├── ds_pipe/                                -> data science pipeline
│   ├── __init__.py
│   ├── __pycache__/
│   ├── data_feed.py                        -> Step 1: data feed
│   ├── data_preprocessing_mp.py            -> Step 2: data preprocessing multiprocessing
│   ├── data_preprocessing.py               -> Step 2: data preprocessing
│   ├── data_research.py                    -> Step 3: data research
│   └── data_results.py                     -> Step 4: data results
├── ds_versions/                            -> ds pipeline versions storage
├── env.txt                                 -> environment variables example
├── features/                               -> features folder: the output of the data science pipeline
│   └── challenge_edMachina/
├── main_ds.py                              -> main data science pipeline script
├── main_ml.py                              -> main machine learning pipeline script
├── Makefile                                -> Makefile script
├── ml_pipe/                                -> machine learning module
│   ├── __init__.py
│   ├── __pycache__/
│   ├── ml_model_1_feature_selector.py      -> Model 1: feature selector (LinearRegression, Lasso, ElasticNet)
│   ├── ml_model_2_random_forest.py         -> Model 2: random forest
│   ├── ml_model_3_cross_validation.py      -> Model 3: cross validation
│   └── ml_model_4_xgboost.py               -> Model 4: xgboost XGBRegressor
├── mlartifacts/                            -> machine learning artifacts folder
│   └── ...
├── mlruns/                                 -> mlflow folder
│   ├── .trash/
│   └── 0/
│       └── ...
├── README.md                               -> README file
├── requirements.txt                        -> requirements file
├── reports/                                -> reports folder (empty)
├── tools/                                  -> tools package
│   ├── __init__.py
│   ├── __pycache__/
│   ├── business.py
│   ├── dates.py
│   ├── ml_plots.py
│   ├── ml.py
│   ├── mlflow.py
│   ├── optimizer.py
│   ├── serializers.py
│   └── sklearn_custom_estimators.py
│   
└── utils/                                  -> utils package
    ├── __init__.py
    ├── __pycache__/
    ├── download_dataset.py                 -> download dataset from Google Drive
    └── config.py                           -> Configuration file
```


#### .4.     **HOW TO RUN DS MAIN PIPELINE**

- Create a virtual environment and install the requirements (once)
```bash
python -m venv venv
source venv/bin/activate
make install
```

- Create the environment variables files (once)
```bash
cp env.txt .env
```

- Conceed administrative permissions to the main script (once)
```bash
chmod +x main_ds.py
```

- Run the main script without reports
```bash
./main_ds.py
```

- Run the main script and download reports
```bash
./main_ds.py -d=True
```



#### .5.     **HOW TO RUN ML TRAIN PIPELINE**

- Conceed administrative permissions to the main script (once)
```bash
chmod +x main_ml.py
```

- Run the main script help command
```bash
./main_ml.py --help
```

- Select the models to run

```bash
./main_ml.py --model_1=True
./main_ml.py --model_2=True --model_3=True
```

- Example of the execution of the models:

![models-command](https://github.com/jackonedev/challenge_edmachina/blob/main/.gif/ml_models-ok.gif?raw=true)

- Example of the help command:

![models-command](https://github.com/jackonedev/challenge_edmachina/blob/main/.gif/ml_help-2-ok.gif?raw=true)

#### .6.     **HOW TO ACCESS MLflow UI**

- For metrics you can access the MLflow UI by running the following command
```bash
mlflow ui
```

#### .7.     **METRICS**     *(Updated)*

- The metrics of the **Linear Regression** models with 'k' features using `sklearn` `SelectKBest`. The models are LinearRegression, Lasso, and ElasticNet.

|    |   k |   train_score_elastic_net |   train_score_lasso |   train_score_lin_reg |   test_score_elastic_net |   test_score_lasso |   test_score_lin_reg |
|---:|----:|--------------------------:|--------------------:|----------------------:|-------------------------:|-------------------:|---------------------:|
|  0 |  34 |                    0.2296 |              0.2846 |                0.296  |                   0.244  |             0.2905 |               0.2959 |
|  1 |  29 |                    0.2114 |              0.2499 |                0.259  |                   0.2303 |             0.2625 |               0.2673 |
|  2 |  25 |                    0.1706 |              0.1881 |                0.1945 |                   0.1786 |             0.1918 |               0.1944 |
|  3 |  23 |                    0.1457 |              0.148  |                0.1501 |                   0.1665 |             0.1726 |               0.1776 |
|  4 |  17 |                    0.1261 |              0.1278 |                0.1291 |                   0.1495 |             0.1549 |               0.1613 |
|  5 |  11 |                    0.1061 |              0.1071 |                0.1077 |                   0.1274 |             0.1292 |               0.1328 |
|  6 |   5 |                    0.0461 |              0.0463 |                0.0464 |                   0.0612 |             0.0616 |               0.0636 |
|  7 |   2 |                    0.006  |              0.0059 |                0.006  |                   0.0063 |             0.0064 |               0.0062 |


<br/><br/>

- The metrics of the **Random Forest** model.

|    | hyperparameters                                           | model_type            |   test_r2_score |   train_r2_score |
|---:|:----------------------------------------------------------|:----------------------|----------------:|-----------------:|
|  0 | {'n_estimators': 100, 'max_depth': 6, 'random_state': 42} | RandomForestRegressor |        0.157031 |         0.370046 |

<br/><br/>

- The metrics of the models using **cross validation** with 5 folds.

|    | hyperparameters                                            | model_type            |   mean_cv_r2_score |   std_cv_r2_score |   test_r2_score |   train_r2_score |
|---:|:-----------------------------------------------------------|:----------------------|-------------------:|------------------:|----------------:|-----------------:|
|  0 | {'alpha': 0.1}                                             | ElasticNet            |           0.205899 |         0.0199696 |        0.243999 |         0.229647 |
|  1 | {'alpha': 0.1}                                             | Lasso                 |           0.260118 |         0.0331183 |        0.290455 |         0.284614 |
|  2 | {}                                                         | LinearRegression      |           0.270514 |         0.0430275 |        0.295887 |         0.295995 |
|  3 | {'n_estimators': 100, 'max_depth': 10, 'random_state': 42} | RandomForestRegressor |           0.204007 |         0.0363636 |        0.223173 |         0.701494 |

<br/><br/>

- The metrics of the **XGBoost** model.

|    | hyperparameters                                                                                                             | model_type   | scaler         |   test_r2_score |   train_r2_score |
|---:|:----------------------------------------------------------------------------------------------------------------------------|:-------------|:---------------|----------------:|-----------------:|
|  0 | {'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.1, 'max_depth': 3, 'callbacks': [EarlyStopping]} | XGBRegressor | StandardScaler |        0.365994 |         0.574565 |



#### .8.     **LAST UPDATES**


**DS MAIN PIPELINE VERSIONING** (August 13)

Add the `next_attempt.sh` bash file for the `new_attempt` step in the Makefile.

The step aims to store the scripts and artifacts related to the current state of the DS Main Pipeline. In this way, it is possible to guarantee the reproducibility of the results, based on older versions of the code, regarding the Data Science workflow phases.

Executing `make new_attempt` will create a folder inside the `ds_versions/` directory, and the folder's name will contain the date and the attempt number, and in it, it will contain a copy of the folders: `ds_pipe`, `features`, and `reports`.



**OPTIMIZATION** (August 2)

The `multiprocessing` library was implemented within the script `tools/optimizer.py`, creating two functions within the script `data_preprocessing_mp.py` that are executed in parallel within the function called `data_preprocessing_mp`.

- The original step 2 takes 7.7 seconds
- The optimized step 2 takes 4.99 seconds
- The original main pipeline takes 10.75 seconds
- The main pipeline with the optimized step takes 8 seconds
- The performance improvement in step 2 is 35.2%
- The performance improvement in the main pipeline is 25.6%
