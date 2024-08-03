# +        __Ed Machina__ 

---

#### .1.     **Ed_Machina_RAW_EDA.ipynb**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16HkTb7CGf8AfLWmzKy9wxFRhyaWFhna_?usp=sharing)


#### .2.     **Ed_Machina_PREPROCESSING_EDA.ipynb**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x7SkJnYftjmRSBVv0NIRDZJCcHluDnYW?usp=sharing)

#### .3.     **OPTIMIZATION**

The `multiprocessing` library was implemented within the script `tools/optimizer.py`, creating two functions within the script `data_preprocessing_mp.py` that are executed in parallel within the function called `data_preprocessing_mp`.

- The original step 2 takes 7.7 seconds
- The optimized step 2 takes 4.99 seconds
- The original main pipeline takes 10.75 seconds
- The main pipeline with the optimized step takes 8 seconds
- The performance improvement in step 2 is 35.2%
- The performance improvement in the main pipeline is 25.6%


#### .4.     **APP DIAGRAM**

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
│   ├── data_research.py                    -> Step 3: data research (not implemented)
│   └── data_results.py                     -> Step 4: data results
├── env.txt                                 -> environment variables example
├── features/                               -> features folder: the output of the data science pipeline
│   └── challenge_edMachina/
├── main_ds.py                              -> main data science pipeline script
├── main_ml.py                              -> main machine learning pipeline script
├── Makefile
├── ml_pipe/                                -> machine learning module
│   ├── __init__.py
│   ├── __pycache__/
│   ├── ml_model_1_feature_selector.py      -> Model 1: feature selector (LinearRegression, Lasso, ElasticNet)
│   ├── ml_model_2_random_forest.py         -> Model 2: random forest
│   └── ml_model_3_cross_validation.py      -> Model 3: cross validation
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


#### .5.     **HOW TO RUN DS MAIN PIPELINE**

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

- Run the main script
```bash
python main_ds.py
``` 

#### .6.     **HOW TO RUN ML MAIN APP**

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

#### .7.     **HOW TO ACCESS MLflow UI**

- For metrics you can access the MLflow UI by running the following command
```bash
mlflow ui
```
