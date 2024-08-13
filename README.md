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



#### .5.     **HOW TO RUN ML MAIN APP**

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

#### .7.     **METRICS**

- The metrics of the **Linear Regression** models with 'k' features using `sklearn` `SelectKBest`. The models are LinearRegression, Lasso, and ElasticNet.

|    |   k |   train_score_elastic_net |   train_score_lasso |   train_score_lin_reg |   test_score_elastic_net |   test_score_lasso |   test_score_lin_reg |
|---:|----:|--------------------------:|--------------------:|----------------------:|-------------------------:|-------------------:|---------------------:|
|  0 |  34 |                    0.2296 |              0.2846 |                0.296  |                   0.244  |             0.2905 |               0.2959 |
|  1 |  29 |                    0.1911 |              0.2207 |                0.2292 |                   0.197  |             0.2212 |               0.2232 |
|  2 |  25 |                    0.1497 |              0.1527 |                0.1552 |                   0.1746 |             0.1816 |               0.1901 |
|  3 |  23 |                    0.1375 |              0.139  |                0.1403 |                   0.1655 |             0.1702 |               0.1774 |
|  4 |  17 |                    0.1251 |              0.1267 |                0.1279 |                   0.1518 |             0.1553 |               0.1617 |
|  5 |  11 |                    0.0668 |              0.067  |                0.0674 |                   0.0891 |             0.0895 |               0.0934 |
|  6 |   5 |                    0.0452 |              0.0454 |                0.0455 |                   0.0587 |             0.059  |               0.0608 |
|  7 |   2 |                    0.0065 |              0.0065 |                0.0065 |                   0.0115 |             0.0114 |               0.012  |


<br/><br/>

- The metrics of the **Random Forest** model.

|    | model_type            |   test_r2_score |   train_r2_score |
|---:|:----------------------|----------------:|-----------------:|
|  0 | RandomForestRegressor |        0.223173 |         0.701494 |

**Hyperparameters:** {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}

<br/><br/>

- The metrics of the models using **cross validation** with 5 folds.

|    | model_type             |   mean_cv_r2_score |   std_cv_r2_score |   test_r2_score |   train_r2_score |
|---:|:-----------------------|-------------------:|------------------:|----------------:|-----------------:|
|  0 | RandomForestClassifier |         -0.03      |         0.055     |        0.258    |                1 |

**Hyperparameters:** {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}

<br/><br/>

- The metrics of the **XGBoost** model.

|    | model_type            |   test_r2_score |   train_r2_score |
|---:|:----------------------|----------------:|-----------------:|
|  0 | XGBRegressor          |        0.366    |         0.575    |

**Hyperparameters:** {"objective": "reg:squarederror", "n_estimators": 1000, "learning_rate": 0.1, "max_depth": 3, "callbacks": [early_stop]}



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
