import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    # Load basic text dataset (good for prototype)
    categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics']
    data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    return X_train_vec, X_test_vec, y_train, y_test

import os

def split_and_save_client_data(X_train, y_train, num_clients=3):
    os.makedirs('client_data', exist_ok=True)
    data_per_client = len(X_train) // num_clients

    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client if i != num_clients - 1 else len(X_train)

        X_client = X_train[start:end]
        y_client = y_train[start:end]

        np.savez(f'client_data/client_{i+1}.npz', X=X_client, y=y_client)

    print(f"✅ Data split into {num_clients} clients and saved to 'client_data/' folder.")

import numpy as np

def load_client_data(client_id):
    """
    Loads client-specific data from the 'client_data/client_{id}.npz' file.
    Returns: (X, y)
    """
    file_path = f"client_data/client_{client_id}.npz"
    data = np.load(file_path)
    return data['X'], data['y']
