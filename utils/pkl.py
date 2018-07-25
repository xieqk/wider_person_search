import pickle

def my_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def my_unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data