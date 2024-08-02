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
