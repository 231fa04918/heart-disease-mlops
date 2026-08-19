import kagglehub
import shutil, os

path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
os.makedirs("data", exist_ok=True)
shutil.copy(os.path.join(path, "heart_disease_uci.csv"), "data/heart.csv")
print("Saved to data/heart.csv")