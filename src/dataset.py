import logging
from pathlib import Path
from typing import Dict, Any, List, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.configuration import StrikeThroughType
from src.cycle_gan import FeatureType


class PairedDataset(Dataset):
    """
    Dataset containing pairs of struck-through words and their clean, ground truth counterparts.
    """

    def __init__(self, rootDir: Path, fold: Union[int, str] = 0, transforms: Compose = None,
                 strokeTypes: Union[List[str], List[StrikeThroughType]] = None, mode: str = "train",
                 featureType: FeatureType = FeatureType.NONE):
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

        if fold == "all":
            if mode == "validation":
                self.fold = -1  # getting rid of "all" so that getitem works properly
                self.struckDir = rootDir / "icdar2021" / mode / "struck"
            else:
                self.struckDir = rootDir / "generated" / mode
            self.groundTruthDir = rootDir.parent / "Dracula_real" / mode / "struck_gt"

        elif fold >= 0:
            if mode == "validation":
                self.struckDir = rootDir / "icdar2021" / mode / "struck"
            else:
                self.struckDir = rootDir / "generated" / mode / "fold_{}".format(fold)
            self.groundTruthDir = rootDir.parent / "Dracula_real" / mode / "struck_gt"
        else:
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
        self.featureType = featureType
        if self.featureType != FeatureType.NONE:
            logger = logging.getLogger("st_removal")
            logger.warning("Selected FeatureType not supported in this training setting, defaulting to FeatureType.NONE")
            self.featureType = FeatureType.NONE

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
        gtImage = Image.open((self.groundTruthDir / entry.image_id).with_suffix(".png"))

        width, height = struckImage.size

        if self.transforms:
            struckImage = self.transforms(struckImage)
            gtImage = self.transforms(gtImage)

        return {"struck": struckImage, "groundTruth": gtImage, "strokeType": entry["strike_type"],
                "path": entry.image_id, "image_size": (height, width)}


class TestDataset(Dataset):
    """
    Dataset containing unrelated struck word images and their related clean ground truth version.
    """

    def __init__(self, rootDir: Path, transforms: Compose = None,
                 strokeTypes: Union[List[StrikeThroughType], List[str]] = None, featureType: FeatureType = FeatureType.NONE):
        """
        Parameters
        ----------
        rootDir : Path
            root dir for the respective data folder
        transforms : Compose
            transformations to apply before returning the respective images
        strokeTypes : Union[List[str], List[StrikeThroughType]]
            stroke types by which to filter the data, i.e. exclude all images with strokes that are not in this list
        """
        self.rootDir = Path(rootDir)
        self.struckDir = rootDir / "struck"
        self.struckGTDir = rootDir / "struck_gt"
        self.struckFileNames = []
        self.struckGTFileNames = []
        self.fileStrokeType = []
        if not strokeTypes:
            self.strokeTypes = ["all"]
        else:
            self.strokeTypes = strokeTypes
        self.featureType = featureType

        csvGlob = list(self.struckDir.glob("*.csv"))
        if len(csvGlob) > 1:
            logger = logging.getLogger("st_removal")
            logger.warning("found more than one csv in train dir; using the first one")
        if len(csvGlob) < 1:
            raise FileNotFoundError("no csv file found in stroke directory")
        csvFile = csvGlob[0]
        struckDf = pd.read_csv(csvFile, dtype={'image_id': str, 'writer_id': str, 'strike_type': str})
        if "all" in self.strokeTypes:
            self.struckFileNames = [(self.struckDir / row.image_id).with_suffix(".png") for row in struckDf.iloc]
            self.fileStrokeType = [
                ((self.struckDir / row.image_id).with_suffix(".png"), StrikeThroughType[row.strike_type]) for row in
                struckDf.iloc]
        else:
            hasCorrectType = struckDf["strike_type"].isin([stroke.name for stroke in self.strokeTypes])
            selectedRows = struckDf[hasCorrectType]
            self.struckFileNames = [(self.struckDir / row.image_id).with_suffix(".png") for row in selectedRows.iloc]
            self.fileStrokeType = [
                ((self.struckDir / row.image_id).with_suffix(".png"), StrikeThroughType[row.strike_type]) for row in
                selectedRows.iloc]

        for fileName in self.struckFileNames:
            name = fileName.name
            self.struckGTFileNames.append(self.struckGTDir / name)

        self.count = len(self.struckFileNames)
        self.transforms = transforms

    def __len__(self) -> int:
        return self.count

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
        struckImage = Image.open(self.struckFileNames[index]).convert('RGB')
        struckImageGroundTruth = Image.open(self.struckGTFileNames[index]).convert('RGB')
        strokeType = self.fileStrokeType[index][1].name
        width, height = struckImage.size

        if self.transforms:
            struckImage = self.transforms(struckImage)
            struckImageGroundTruth = self.transforms(struckImageGroundTruth)

        return {'struck': struckImage, 'groundTruth': struckImageGroundTruth, 'strokeType': strokeType,
                'path': str(self.struckFileNames[index].name), "image_size": (height, width)}
