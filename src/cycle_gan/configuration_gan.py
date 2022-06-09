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
from typing import List, Tuple, Union

import torch

from src.configuration import DatasetChoice, StrikeThroughType


class ExperimentType(Enum):
    """
    Encodes the type of experiment.
    """
    ORIGINAL = 0
    FEATURE_RECOG = 1
    STROKE_RECOG = 2
    NO_RECOG = 3

    @classmethod
    def getByName(cls, name: str) -> "ExperimentType":
        """
        Returns the ExperimentType corresponding to the given name.

        Parameters
        ----------
        name : str
            string that should be converted to a ExperimentType

        Returns
        -------
        ExperimentType
            ExperimentType representation of the provided string

        Raises
        ------
        ValueError
            if the given name does not correspond to a ExperimentType
        """
        name = name.upper()
        for t in ExperimentType:
            if t.name == name:
                return t
        raise ValueError("Unknown experiment type " + name)


class FeatureType(Enum):
    """
    Encodes the type of features to be appended to the input image.
    """
    NONE = 0
    SCALAR = 1
    TWO_CHANNEL = 2
    CHANNEL = 3
    CHANNEL_RANDOM = 4
    RANDOM = 5

    @classmethod
    def getByName(cls, name: str) -> "FeatureType":
        """
        Returns the FeatureType corresponding to the given name.

        Parameters
        ----------
        name : str
            string that should be converted to a FeatureType

        Returns
        -------
        FeatureType
            FeatureType representation of the provided string

        Raises
        ------
        ValueError
            if the given name does not correspond to a FeatureType
        """
        name = name.upper()
        for t in FeatureType:
            if t.name == name:
                return t
        raise ValueError("Unknown feature type " + name)


class ModelName(Enum):
    """
    Encodes the names of supported base models.
    """
    DENSE = auto()
    RESNET = auto()

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
            return ModelName.RESNET


class ConfigurationGAN:
    """
    Holds the configuration for the current experiment.
    """

    def __init__(self, parsedConfig: SectionProxy, test: bool = False, output_dir=None, fileSection: str = "ORIGINAL",
                 train_dataset=None, test_dataset=None, batchSize=None, identityLambda=None, cleanLambda=None,
                 struckLambda=None):
        self.fileSection = fileSection
        if train_dataset is not None:
            self.train_dataset_choice = train_dataset
            self.test_dataset_choice = test_dataset
        else:
            self.train_dataset_choice = parsedConfig.get('dataset_choice_train')
            self.test_dataset_choice = parsedConfig.get('dataset_choice_test')

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.epochs = parsedConfig.getint('epochs', 100)
        self.learningRate = parsedConfig.getfloat('learning_rate', 0.0002)
        self.betas = self.parseBetas(parsedConfig.get("betas", "0.5,0.999"))
        self.batchSize = batchSize if batchSize else parsedConfig.getint('batchsize', 4)
        self.imageHeight = parsedConfig.getint('imageheight', 128)
        self.imageWidth = parsedConfig.getint('imagewidth', 256)
        self.modelSaveEpoch = parsedConfig.getint('modelsaveepoch', 10)
        self.validationEpoch = parsedConfig.getint('validation', 10)

        self.dataset_dir = parsedConfig.get('dataset_dir', 'datasets')
        full_dataset_path = self.dataset_dir + "/"
        self.trainImageDir = Path(
            full_dataset_path + self.train_dataset_choice)
        self.testImageDir = Path(
            full_dataset_path + self.test_dataset_choice)
        self.invertImages = parsedConfig.getboolean('invert_images', False)

        self.blockCount = parsedConfig.getint('blockcount', 6)

        self.poolSize = parsedConfig.getint('poolsize', 50)

        trainTypes = parsedConfig.get('train_stroke_types', 'all')
        self.trainStrokeTypes = self.parseStrokeTypes(trainTypes)

        testTypes = parsedConfig.get('test_stroke_types', '')
        self.testStrokeTypes = self.parseStrokeTypes(testTypes)
        if len(self.testStrokeTypes) < 1:
            self.testStrokeTypes = self.trainStrokeTypes

        self.count = parsedConfig.getint('count', 1000000)
        self.validationCount = parsedConfig.getint('val_count', 1000000)
        self.experiment = ExperimentType.getByName(parsedConfig.get('experiment', "original"))
        self.featureType = FeatureType.getByName(parsedConfig.get("featureType", "none"))
        if self.featureType != FeatureType.NONE and self.experiment == ExperimentType.ORIGINAL:
            self.featureType = FeatureType.NONE
            parsedConfig['featureType'] = "NONE"

        self.padScale = parsedConfig.getboolean('padscale', False)
        self.padWidth = parsedConfig.getint('padwidth', 512)
        self.padHeight = parsedConfig.getint('padheight', 128)
        self.cnnLambda = parsedConfig.getfloat('cnn_lambda', 0.5)
        self.identityLambda = identityLambda if identityLambda is not None else parsedConfig.getfloat('identity_lambda', 0.5)
        self.cleanLambda = cleanLambda if cleanLambda is not None else parsedConfig.getfloat('clean_lambda', 10.0)
        self.struckLambda = struckLambda if struckLambda is not None else parsedConfig.getfloat('struck_lambda', 10.0)
        print(f"Lambda_{self.identityLambda}_{self.cleanLambda}_{self.struckLambda}")
        self.discWithFeature = parsedConfig.getboolean('disc_feature', False)
        if self.featureType == FeatureType.NONE:
            self.discWithFeature = False

        if not test:
            output_dir = output_dir if output_dir else parsedConfig.get('outdir')
            self.outDir = Path(output_dir) / '{}_{}_{}_{}_{}'.format(fileSection, str(int(time.time())),
                                                                     random.randint(0, 100000),
                                                                     self.train_dataset_choice, self.batchSize)
            parsedConfig['outdir'] = str(self.outDir)
            print(output_dir)

        if not test and not self.outDir.exists():
            self.outDir.mkdir(parents=True, exist_ok=True)

        f = parsedConfig.get("fold")
        if f == "all" and self.train_dataset_choice == DatasetChoice.Dracula_synth.name or self.train_dataset_choice == DatasetChoice.Dracula_synth.name:
            self.fold = "all"
            parsedConfig["fold"] = "all"
        else:
            self.fold = f
            if self.fold != -1 and self.train_dataset_choice != DatasetChoice.Dracula_synth.name:
                self.fold = -1

        self.modelName = ModelName.getByName(parsedConfig.get("model", "RESNET"))

        if not test:
            configOut = self.outDir / 'config.cfg'
            with configOut.open('w+') as cfile:
                parsedConfig.parser.write(cfile)

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


def getDynamicConfigurationGAN(configSection, configFile, output_dir=None, train_dataset=None, test_dataset=None,
                               dynamic=True, batchSize=None, identityLambda=None, cleanLambda=None,
                               struckLambda=None) -> ConfigurationGAN:
    fileSection = 'ORIGINAL'
    if dynamic:
        fileName = 'config_files/serverConfig_strike_rem_dyn.cfg'
    else:
        fileName = 'config_files/serverConfig_strike_rem.cfg'
    if configSection:
        fileSection = configSection
    if configFile:
        fileName = configFile
    configParser = configparser.ConfigParser()
    configParser.read(fileName)
    parsedConfig = configParser[fileSection]
    sections = configParser.sections()
    for s in sections:
        if s != fileSection:
            configParser.remove_section(s)
    if train_dataset is not None and test_dataset is not None:
        return ConfigurationGAN(parsedConfig, False, None, fileSection, train_dataset,
                                test_dataset, batchSize, identityLambda, cleanLambda, struckLambda)
    else:
        return ConfigurationGAN(parsedConfig, fileSection=fileSection)


def getConfigurationGAN() -> ConfigurationGAN:
    """
    Reads the required arguments from command line and parse the respective configuration file/section.

    Returns
    -------
        parsed :class:`Configuration`
    """
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-config", required=False, help="section of config-file to use")
    cmdParser.add_argument("-configfile", required=False, help="path to config-file")
    args = vars(cmdParser.parse_args())
    return getDynamicConfigurationGAN(args["config"], args['configfile'], dynamic=True, batchSize=8)
