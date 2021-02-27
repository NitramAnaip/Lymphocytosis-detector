import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LymphocytosisDataset(Dataset):
    # nb_img_fill represents the a common maximum for
    nb_img_fill = 200

    def __init__(
        self,
        annotation_csv: str,
        root_dir: str,
        train: bool = True,
        valid: bool = False, #If we want the validation set. train must be set at True to get valiation set
        train_split: float, # percentage we want in train (as opposed to valdation)
        transform: callable = None,
        fill_img_list: bool = False,
        split_label: bool = False,
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
        ## Identify training, validation, and testing ids
        self.train_ids = list(
            self.annotation_frame["ID"][self.annotation_frame["LABEL"] != -1]
        )

        if valid:
            self.train_ids = self.train_ids[:int(len(self.train_ids) * train_split)]
            self.valid_ids = self.train_ids[int(len(self.train_ids) * train_split):]


        self.test_ids = list(
            self.annotation_frame["ID"][self.annotation_frame["LABEL"] == -1]
        )
        ## Reindex the annotation to the "ID" column
        self.annotation_frame = self.annotation_frame.set_index("ID")
        ## Handle "DOB" column
        DOB = self.annotation_frame["DOB"].copy()
        line_with_slash = DOB.str.contains("/")
        DOB[line_with_slash] = pd.to_datetime(DOB[line_with_slash], format="%m/%d/%Y")
        DOB[~line_with_slash] = pd.to_datetime(DOB[~line_with_slash], format="%d-%m-%Y")
        self.annotation_frame["DOB"] = pd.to_datetime(DOB)

        ## Define instance attributes
        self.root_dir = root_dir
        self.train = train
        self.valid = valid
        self.transform = transform
        self.fill_img_list = fill_img_list
        self.split_label = split_label

    def __len__(self) -> int:
        if self.train:
            if self.valid:
                return int( len(self.train_ids) * (1-self.train_split) )
            else:
                return int(len(self.train_ids) * self.train_split)
        else:
            return len(self.test_ids)

    def __getitem__(self, index: int) -> tuple:
        if self.train:
            if self.valid:
                id = self.valid_ids[index]
            else:
                id = self.train_ids[index]
        else:
            id = self.test_ids[index]
        annotation = self.annotation_frame.loc[id]
        img_dir = f"{self.root_dir}/{'train' if self.train else 'test'}set/{id}/"
        img_list = [
            Image.open(os.path.join(img_dir, img_name))
            for img_name in os.listdir(img_dir)
        ]
        if self.fill_img_list:
            missing = self.nb_img_fill - len(img_list)
            filling_indices = np.random.randint(len(img_list), size=missing)
            for index in filling_indices:
                img_list.append(img_list[index])
        if self.split_label:
            label = annotation["LABEL"]
            annotation = annotation.drop("LABEL")
            return annotation, img_list, label
        return annotation, img_list
