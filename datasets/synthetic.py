from sklearn.datasets import make_classification
from .base_dataset import Dataset

import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def get_dataset(self, **kwargs):
        n_features = kwargs.get("n_features")
        rmv_features = kwargs.pop("remove")
        rmv_features = [int(x) for x in rmv_features]

        np_x, np_y = make_classification(**kwargs)
        np_y = np_y.reshape((-1, 1))

        np_x = np.delete(np_x, rmv_features, 1)

        self.set_metadata(self.default_metadata(range(n_features)))

        return np_x, np_y
    
    # def get_metadata(self, rmv_features):
    #     metadata = {x: (x, x) for x in range(self.n_features)}
    #     rmv_features = [int(x) for x in rmv_features]
    #     rmv_size = 0
    #     for k, v in metadata.items():
    #         if k in rmv_features:
    #             rmv_size += 1
    #         else:
    #             metadata[k] = (v[0]-rmv_size, v[1]-rmv_size)
    #     return metadata