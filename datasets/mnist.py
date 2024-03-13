import os
import numpy as np
import pandas as pd
from base_dataset import ImageDataset

FILENAME = lambda set, type : os.path.join(os.path.abspath(''), "mnist", f"mnist-{set}-{type}.npy")

class MnistDataset(ImageDataset):
    @staticmethod
    def get_df_data() -> pd.DataFrame:
        img_train = list(np.load(FILENAME("train", "images")))
        lbl_train = np.load(FILENAME("train", "labels"))

        data = pd.DataFrame(data={"labels": lbl_train, "images": img_train}, dtype=object)
        data['labels'] = data["labels"].apply(str)
        return data
    
    @staticmethod
    def get_label_column() -> str:
        return "labels"