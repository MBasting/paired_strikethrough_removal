import logging
from pathlib import Path
from typing import Dict, Any, List, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.configuration import StrikeThroughType


class BigPairedDataset(Dataset):
    """
    Dataset containing pairs of struck-through words and their clean, ground truth counterparts.
    """

    def __init__(self, rootDir: Path, fold: Union[int, str] = 0, transforms: Compose = None,
                 strokeTypes: Union[List[str], List[StrikeThroughType]] = None, mode: str = "train"):
        """
        Parameters
        ----------
        rootDir : Path
            root dir for the respective data folder
        fold : Union[int, str]
            fold number between 0 and 5, or 'all' for accumulation over all five folds or -1 for datasets without folds
        transforms : Compose
            transformations to apply before returning the respective images
        strokeTypes : Union[List[str], List[StrikeThroughType]]
            stroke types by which to filter the data, i.e. exclude all images with strokes that are not in this list
        mode : str
            'train' or 'validation' - which subdirectory to load the data from
        """
        self.transforms = transforms
        self.fold = fold
        self.struckDir = rootDir / mode / "struck"
        self.groundTruthDir = rootDir / mode / "struck_gt"



        if fold == "all" and mode != "validation":
            foldData = []
            for foldIndex in range(5):
                foldDir = rootDir / "generated" / mode / "fold_{}".format(foldIndex)
                data = pd.read_csv(foldDir / "struck_{}.csv".format(mode), dtype={"image_id": str})
                data['fold_image_id'] = "fold_{}/".format(foldIndex) + data.image_id
                foldData.append(data)
            self.data = pd.concat(foldData)
        else:
            self.data = pd.read_csv(self.struckDir / "struck_{}.csv".format(mode), dtype={"image_id": str})
        if strokeTypes and "all" not in strokeTypes:
            self.data = self.data[self.data["strike_type"].isin(strokeTypes)].reset_index()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
            - "struck": the struck image
            - "groundTruth": the ground truth image
            - "strokeType": the type of strikethrough applied to the image
            - "path": the name of the image
            - "image_size": image dimensions (height, width)

        Parameters
        ----------
        index : int
            index at which to retrieve the image

        Returns
        -------
        Dict[str, Any]
            dictionary containing the aforementioned pieces of information
        """
        entry = self.data.iloc[index]

        if self.fold == "all":
            struckImage = Image.open((self.struckDir / entry.fold_image_id).with_suffix(".png"))
        else:
            struckImage = Image.open((self.struckDir / entry.image_id).with_suffix(".png"))

        gtImage = Image.open((self.groundTruthDir / entry.image_id.split("_")[0]).with_suffix(".png"))

        width, height = struckImage.size

        if self.transforms:
            struckImage = self.transforms(struckImage)
            gtImage = self.transforms(gtImage)

        return {"struck": struckImage, "groundTruth": gtImage, "strokeType": entry["strike_type"],
                "path": entry.image_id, "image_size": (height, width)}

