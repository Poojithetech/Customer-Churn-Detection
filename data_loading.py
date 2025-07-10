# data_loading.py

import pandas as pd
from config import TRAIN_DATA_PATH, TEST_DATA_PATH

def load_data():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    return train_df, test_df

# For testing the loading step
if __name__ == "__main__":
    train, test = load_data()
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
