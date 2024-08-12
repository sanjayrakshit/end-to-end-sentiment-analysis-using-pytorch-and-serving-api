# Sentiment Analysis on Movie Reviews

Specific packages required can be found in [requirements.txt](./requirements.txt). Other packages are genereic ones like
numpy and pandas etc.

Dataset: [IMDB large dataset](data/imdb/IMDB%20Dataset.csv)

Exploratory Data Analysis and base model: [EDA + base models.ipynb](./EDA%20+%20base%20models.ipynb) (Some of the
findings are here)

## RNN based technique

Model was trained on google colab. Here are the [training logs](./colab.log). Here is
the [training notebook](./trainer_sentiment_analysis.ipynb)

Best model was found at `Epoch - 55`

```
              precision    recall  f1-score   support

           0       0.88      0.83      0.85      1432
           1       0.83      0.88      0.86      1419

    accuracy                           0.85      2851
   macro avg       0.85      0.85      0.85      2851
weighted avg       0.85      0.85      0.85      2851
```

### Overview

1. First is data preparation and preprocessing.
    - For this we use [dataset_preparation.py](./dataset_preparation.py). Modify the `dataset_path` according to your
      requirement.
    - Select required sample size `df = df.sample(n=15000)`
    - Give appropriate name to save it. `save_as_pickle(train_df, 'train_15K.pkl')
      save_as_pickle(test_df, 'test_15K.pkl')
2. Then [data_loaders.py](./data_loaders.py). We can change the `MAX_LEN=512` to our required specification. I've found
   512 to be okay, but it makes the training slower.I've tried with 100, but it doesn't work that well. Here we are
   creating a `torch.Dataset` class which would be used with `torch.Dataloaders` for batch training.
3. Next is [models.py](./models.py). Here, I've defined the actual model `LSTMModel`. The architecture is as follows
    * ___Embedding -> Bi-directional LSTM -> Bi-directional LSTM -> 1st time sequence output -> Dense -> Dense ->
      Output___
4. Next is [trainer.py](./trainer.py). Here I've defined alot of utility function for abstracting the trainer code. But
   the sequence of steps goes like this:
    1. Load the train and validation dataset using `dataset` and `dataloader`
    2. Initialize model and load into available `device`
    3. Define `optimizer`
    4. Run `epoch`
        1. Train over batches *(forward + loss + backward + weight update)* and accumulate loss
        2. Run eval over eval batches *(forward + loss)* and accumulate loss
        3. Save if gets better or Halt epoch if model doesn't get better after certain amount of time.

We would be serving this LSTM based model

## Serving

For serving the model we have defined an endpoint which works on `0.0.0.0:8001`

We do it with the help of [predictor_endpoint.py](./predictor_endpoint.py). Inside the `predict_sentiment()` we are
getting the text from the payload and then preprocessing it using the same fucntion we preprocessed our train data with.

To serve we can run this command

```shell
uvicorn predictor_endpoint:app --host 0.0.0.0 --port 8001 --reload
```

OR

```shell
/start_endpoint.sh
```

It does the same thing.

Then the request should be in the format as follows

```json
{
  "text": "This is the most amazing movie I've ever seen."
}
```

And the response should be expected like

```json
{
  "review": "This is the most amazing movie I've ever seen.",
  "score": 0.9959127306938171,
  "tag": "Great"
}
```

### Explanation about the scoring trick in serving

The original problem was to categorize reviews into 4 classes, Great, Good, Bad or Worst. However, the dataset I had
choosen had only binary class setting. To make it work, I had to utilize the score that is given by the last sigmoid
layer. The logic is as follows:

```python
tag = None
if score <= 0.25:
    tag = 'Worst'
elif score <= 0.50:
    tag = 'Bad'
elif score <= 0.75:
    tag = 'Good'
else:
    tag = 'Great'
```

# Future work

* Base models with
    * W2V
    * tf-idf weighted W2V
    * Fasttext for OOV
    * Pre-trained BERT embeddings
* Bert model (frozen + finetuning) with some dense layers
* Error analysis on misclassified points
* Thresholding on sigmoid scores in case of class imbalance, but here we had balanced dataset
* Calibration to get actual probability
* Train with entire dataset or with a dataset which has proper classes
* n-gram as a feature with `good to bad ratio` and `bad to good ratio`
  . [`good to bad ratio` - Total n-grams where we have a preceeding positve word and succeding negative work.`bad to good ratio` - Total n-grams where we have a preceeding negative word and succeding positive word]
* Ensembling base models (also ensembling DL models given enough infrastructure)