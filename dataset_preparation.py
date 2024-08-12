import pickle

import multiprocess as mp
import numpy as np
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# Loading spacy en_core_web_sm model. We will be using it's lemmatizer
nlp = spacy.load('en_core_web_sm')


def load_dataframe_from_csv(path):
    """
    Function that loads the dataframe
    :param path: path to load the dataframe from
    :return: Loaded dataframe
    """
    return pd.read_csv(path)


def split_data(data, test_size, stratify_column):
    """
    Fucntion that splits the data into train and test
    :param data: Actual dataframe object
    :param test_size: The size of the test_data
    :param stratify_column: The column to stratify with
    :return: Splitted data
    """
    return train_test_split(data, test_size=test_size, stratify=data[stratify_column], random_state=42)


def clean_and_lemmatize(text):
    """
    A function that cleans the html tags and lemmatizes the text
    :param text: Text to clean
    :return: Cleaned text
    """
    if text.strip():
        text = BeautifulSoup(text, "lxml").text
        doc = nlp(text)
        lemmatized_list = [token.lemma_ for token in doc]
        return ' '.join(lemmatized_list)
    else:
        return ''


def parallelize(series, func, cores=None):
    """
    A function to pararllelize the process of apply on a series
    :param series:
    :param func: The function to apply
    :param cores: Number of cores in cpu
    :return: Modified series
    """
    if not cores or cores > mp.cpu_count():
        cores = mp.cpu_count() - 1
    series_split = np.array_split(series, cores)
    pool = mp.Pool(cores)
    modified_series = pd.concat(pool.map(lambda x: x.apply(func), series_split))
    pool.close()
    pool.join()
    return modified_series


def string_len(text):
    """
    Returns the length of the text
    :param text: The text
    :return: len
    """
    return len(text)


def words_in_sentence(text):
    """
    Finds the number of words
    :param text: The text
    :return: The number of words
    """
    return len(text.split())


# def tokenizer_with_sentiment(text, label):
#     text = [i[:510] for i in text]
#     tokenized = tokenizer(text)
#     tokenized['labels'] = label
#     return tokenized


def binarize_labels(label):
    """
    A function to binarize the labels
    :param label: Label as list of text
    :return: 1/0 label
    """
    return 1 if label == 'positive' else 0


def save_as_pickle(data, path):
    """
    Pickling function
    :param data: data to pickle
    :param path: Path to save
    :return: None
    """
    with open(path, 'wb') as wr:
        pickle.dump(data, wr)


if __name__ == '__main__':
    dataset_path = 'data/imdb/IMDB Dataset.csv'
    df = load_dataframe_from_csv(dataset_path)

    # Taking random 15K points out of 50K
    df = df.sample(n=15000)

    # This will take time
    print('Wait this will take few minutes ...')
    df['review'] = parallelize(df['review'], clean_and_lemmatize)
    df['total_length'] = df['review'].apply(string_len)
    df['total_length_in_words'] = df['review'].apply(words_in_sentence)
    df['sentiment'] = df['sentiment'].apply(binarize_labels)

    # There were some odly large reviews. We are excluding them
    threshold = np.percentile(df['total_length_in_words'], 95)
    df = df[df['total_length_in_words'] <= threshold]

    # Splitting
    train_df, test_df = split_data(df, 0.2, 'sentiment')

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Saving
    save_as_pickle(train_df, 'train_15K.pkl')
    save_as_pickle(test_df, 'test_15K.pkl')
