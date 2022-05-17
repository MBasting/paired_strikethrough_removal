"""
Contains all code related to the configuration of experiments.
"""
import argparse
import configparser
import random
import time
from configparser import SectionProxy
from enum import Enum, auto
from pathlib import Path
from typing import Tuple, List, Union

import torch


class StrikeThroughType(Enum):
    """
    Encodes the type of strikethrough
    """
    SINGLE_LINE = 0
    DOUBLE_LINE = 1
    DIAGONAL = 2
    CROSS = 3
    WAVE = 4
    ZIG_ZAG = 5
    SCRATCH = 6

    @classmethod
    def value_for_name(cls, name):
        for st in StrikeThroughType:
            if st.name == name:
                return st.value
        raise ValueError("Unknown name {}".format(name))

    def as_list(self):
        return [x.name for x in StrikeThroughType]


class ModelName(Enum):
    """
    Encodes the names of supported base models.
    """
    DENSE = auto()
    CNN = auto()
    SHALLOW = auto()
    TIRAMISU = auto()

    @staticmethod
    def getByName(name: str) -> "ModelName":
        """
        Returns the ModelName corresponding to the given string. Returns ModelName.RESNET in case an unknown name is
        provided.

        Parameters
        ----------
        name : str
            string representation that should be converted to a ModelName

        Returns
        -------
            ModelName representation of the provided string, default: ModelName.RESNET
        """
        if name.upper() in [model.name for model in ModelName]:
            return ModelName[name.upper()]
        else:
            return ModelName.DENSE


class FileSection(Enum):
    """
    Encodes the names of supported sections
    """
    DEFAULT = auto()
    SIMPLE_CNN = auto()
    SHALLOW = auto()
    UNET = auto()
    GENERATOR = auto()

    @staticmethod
    def getByName(name: str) -> "FileSection":
        """
        Returns the FileSection corresponding to the given string. Returns FileSection.RESNET in case an unknown name is
        provided.

        Parameters
        ----------
        name : str
            string representation that should be converted to a FileSection

        Returns
        -------
            FileSection representation of the provided string, default: FileSection.DEFAULT
        """
        if name.upper() in [section.name for section in FileSection]:
            return FileSection[name.upper()]
        else:
            return FileSection.DEFAULT


class DatasetChoice(Enum):
    IAMsynth_full = auto()
    Dracula_real = auto()
    Dracula_synth = auto()

    @staticmethod
    def getByName(name: str) -> str:
        if name in [ds_choice.name for ds_choice in DatasetChoice]:
            return DatasetChoice[name]
        else:
            return DatasetChoice.IAMsynth_full


class Configuration:
    """
    Holds the configuration for the current experiment.
    """

    def __init__(self, parsedConfig: SectionProxy, test: bool = False, fileSection: str = "DEFAULT", train_dataset: str = "IAMsynth_full", test_dataset: str = "IAMsynth_full"):
        self.parsedConfig = parsedConfig
        self.fileSection = FileSection.getByName(fileSection).name
        self.train_dataset_choice = train_dataset
        self.test_dataset_choice = test_dataset
        if not test:
            self.outDir = Path(self.parsedConfig.get('out_dir')) / '{}_{}_{}_{}_{}'.format(
                self.getSetStr("model", "DENSE"),
                fileSection,
                str(int(time.time())),
                random.randint(0, 100000),
                self.train_dataset_choice)
            self.parsedConfig['out_dir'] = str(self.outDir)

        if not test and not self.outDir.exists():
            self.outDir.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.epochs = self.getSetInt("epochs", 100)
        self.learningRate = self.getSetFloat("learning_rate", 0.001)
        self.betas = self.parseBetas(self.getSetStr("betas", "0.9,0.999"))
        self.finetune = self.getSetBoolean("fine_tune", False)

        self.batchSize = self.getSetInt('batch_size', 4)
        self.imageHeight = self.getSetInt('image_height', 128)
        self.imageWidth = self.getSetInt('image_width', 256)
        self.modelSaveEpoch = self.getSetInt('model_save_epoch', 10)
        self.validationEpoch = self.getSetInt('validation_epoch', 10)
        self.dataset_dir = self.getSetStr('dataset_dir', "datasets")
        full_dataset_path = self.dataset_dir + "/"
        self.trainImageDir = Path(
            full_dataset_path + self.train_dataset_choice)
        self.testImageDir = Path(
            full_dataset_path + self.test_dataset_choice)
        self.invertImages = self.getSetBoolean('invert_images', False)
        self.greyscale = self.getSetBoolean('greyscale', False)
        self.blockCount = self.getSetInt('block_count', 1)

        trainTypes = self.getSetStr('train_stroke_types', 'all')
        self.trainStrokeTypes = self.parseStrokeTypes(trainTypes)

        testTypes = parsedConfig.get('test_stroke_types', '')
        if not testTypes:
            self.getSetStr("test_stroke_types", parsedConfig.get('train_stroke_types'))
            self.testStrokeTypes = self.trainStrokeTypes
        else:
            self.testStrokeTypes = self.parseStrokeTypes(testTypes)

        self.loss = self.getSetStr("loss", "bce")

        self.padScale = self.getSetBoolean('pad_scale', False)
        self.padWidth = self.getSetInt('pad_width', 512)
        self.padHeight = self.getSetInt('pad_height', 128)

        self.down = self.parseTiramisuConfig(self.getSetStr("down", "4"))
        self.bottleneck = self.getSetInt("bottleneck", 4)
        self.up = self.parseTiramisuConfig(self.getSetStr("up", "4"))

        f = parsedConfig.get("fold")
        if f == "all" and self.train_dataset_choice == DatasetChoice.Dracula_synth.name:
            self.fold = f
            self.parsedConfig["fold"] = "all"
        else:
            self.fold = self.getSetInt("fold", -1)
            if self.fold != -1 and self.train_dataset_choice != DatasetChoice.Dracula_synth.name:
                self.fold = -1

        self.modelName = ModelName.getByName(self.getSetStr("model", "DENSE"))

        if not test:
            configOut = self.outDir / 'config.cfg'
            with configOut.open('w+') as cfile:
                parsedConfig.parser.write(cfile)

    def getSetInt(self, key: str, default: int = None):
        value = self.parsedConfig.getint(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetFloat(self, key: str, default: float = None):
        value = self.parsedConfig.getfloat(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetBoolean(self, key: str, default: bool = None):
        value = self.parsedConfig.getboolean(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetStr(self, key: str, default: str = None):
        value = self.parsedConfig.get(key, default)
        self.parsedConfig[key] = str(value)
        return value

    @staticmethod
    def parseBetas(betaString: str) -> Tuple[float, float]:
        """
        Parses a comma-separated string to a list of floats.

        Parameters
        ----------
        betaString: str
            String to be parsed.

        Returns
        -------
            Tuple of floats.

        Raises
        ------
        ValueError
            if fewer than two values are specified
        """
        betas = betaString.split(',')
        if len(betas) < 2:
            raise ValueError("found fewer than two values for betas")
        return float(betas[0]), float(betas[1])

    @staticmethod
    def parseTiramisuConfig(configString: str) -> List[int]:
        split = configString.split(",")
        result = [int(s.strip()) for s in split]
        return result

    @staticmethod
    def parseStrokeTypes(strokeString: str) -> Union[List[StrikeThroughType], List[str]]:
        """
        Parses a comma-separated string to a list of stroke types.

        Parameters
        ----------
        strokeString : str
            string to be parsed

        Returns
        -------
        List[StrikeThroughType]
            list of stroke type strings

        """
        if '|' in strokeString:
            splitTypes = strokeString.split('|')  # for backward compatibility
        else:
            splitTypes = strokeString.split(',')
        strokeTypes = []
        if "all" in splitTypes:
            strokeTypes = ["all"]
        else:
            for item in splitTypes:
                item = item.strip()
                if item in [stroke.name for stroke in StrikeThroughType]:
                    strokeTypes.append(item)  # StrikeThroughType[item].name)
        if len(strokeTypes) < 1:
            strokeTypes = ["all"]
        return strokeTypes


def getConfiguration_dynamic(file=None, section=None, train_dataset=None, test_dataset=None) -> Configuration:
    """

    Returns
    -------
        parsed :class:`Configuration`
    """
    fileSection = 'DEFAULT'
    fileName = 'config_files/config.cfg'
    if section:
        fileSection = section
    if file:
        fileName = file
    configParser = configparser.ConfigParser()
    configParser.read(fileName)
    parsedConfig = configParser[fileSection]
    sections = configParser.sections()
    for s in sections:
        if s != fileSection:
            configParser.remove_section(s)
    if train_dataset is not None and test_dataset is not None:
        return Configuration(parsedConfig, fileSection=fileSection, train_dataset=train_dataset, test_dataset=test_dataset)
    else:
        return Configuration(parsedConfig, fileSection=fileSection)


def getConfiguration() -> Configuration:
    """
    Reads the required arguments from command line and parse the respective configuration file/section.

    Returns
    -------
        parsed :class:`Configuration`
    """
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-section", required=False, help="section of config-file to use")
    cmdParser.add_argument("-file", required=False, help="path to config-file")
    args = cmdParser.parse_args()
    return getConfiguration_dynamic(args.file, args.section)
