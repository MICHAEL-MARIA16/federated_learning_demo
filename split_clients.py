from utils import load_data, split_and_save_client_data

X_train, X_test, y_train, y_test = load_data()
split_and_save_client_data(X_train, y_train, num_clients=3)
