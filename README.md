# ICD9_Deep_Learning

## Google Big Query Setup

* [Using Big Query to Access MIMIC-III](https://mimic.mit.edu/docs/iii/tutorials/intro-to-mimic-iii-bq/)
* [Setting up GCP CLI](https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to)

## Environment Setup
```
conda env create -f environment.yml
conda activate cs6250-finalproject
```

## Data Prep

We will embed the MIMIC-III data before using it while training the model. Be sure to set up GCP Big Query before doing this step. Also, specify the project name you set up in GCP in `data_prep.py`.

```
python data_prep.py
```

## Model Training

Run the `test.ipynb` Jupyter notebook.