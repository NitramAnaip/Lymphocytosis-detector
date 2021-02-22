import os
from os import path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LymphocytosisDataset(Dataset):
    def __init__(
        self,
        annotation_csv: str,
        root_dir: str,
        train: bool = True,
        transform: callable = None,
    ) -> None:
        """
        Loads Lymphocytosis data

        Parameters
        ----------
        annotation_csv : str
            Path to the csv file that contains annotation for every patient
        root_dir : str
            Directory with all the images
        train : bool
            Wether to use training images or not
        transform : callable, optional
            Optional transform to be applied on sample images, by default None
        """
        ## Get the clinical annotation
        self.annotation_frame = pd.read_csv(annotation_csv, index_col=0)
        ## Identify training and testing ids
        self.train_ids = list(
            self.annotation_frame["ID"][self.annotation_frame["LABEL"] != -1]
        )
        self.test_ids = list(
            self.annotation_frame["ID"][self.annotation_frame["LABEL"] == -1]
        )
        ## Reindex the annotation to the "ID" column
        self.annotation_frame = self.annotation_frame.set_index("ID")
        ## Split annotations and labels
        self.labels = self.annotation_frame["LABEL"]
        self.annotation_frame = self.annotation_frame.drop(columns=["LABEL"])
        ## Handle "DOB" column
        DOB = self.annotation_frame["DOB"].copy()
        line_with_slash = DOB.str.contains("/")
        DOB[line_with_slash] = pd.to_datetime(DOB[line_with_slash], format="%m/%d/%Y")
        DOB[~line_with_slash] = pd.to_datetime(DOB[~line_with_slash], format="%d-%m-%Y")
        self.annotation_frame["DOB"] = pd.to_datetime(DOB)

        ## Define instance attributes
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

    def __len__(self) -> int:
        if self.train:
            return len(self.train_ids)
        else:
            return len(self.test_ids)

    def __getitem__(self, index: int) -> tuple:
        if self.train:
            id = self.train_ids[index]
        else:
            id = self.test_ids[index]
        annotation = self.annotation_frame.loc[id]
        img_dir = f"{self.root_dir}/{'train' if self.train else 'test'}set/{id}/"
        img_list = [
            Image.open(os.path.join(img_dir, img_name))
            for img_name in os.listdir(img_dir)
        ]
        label = self.labels.loc[id]
        return annotation, img_list, label
