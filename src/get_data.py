import kagglehub
import shutil
import os

DATA_PATH = "data/raw/heart_disease_v1.csv"

if os.path.exists(DATA_PATH):
    print("Dataset found!")
    print("Path:", DATA_PATH)
else:
    print("Dataset not found!")
    print("Expected path:", DATA_PATH)
