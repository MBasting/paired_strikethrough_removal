import logging
from math import ceil
from typing import Tuple, List

import numpy as np
import torch
from PIL import ImageOps, Image
from models import Tiramisu
from skimage.transform import rescale
from torchvision import transforms

from src import configuration
from src._models import DenseGenerator, Shallow, SimpleCNN
from src.configuration import ModelName


def initLoggers(config: configuration.Configuration, mainLoggerName: str, auxLoggerNames: List[str] = None):
    """
    Utility function initialising a default info logger, as well as several loss loggers.

    Parameters
    ----------
    config : Configuration
        experiment configuration, to obtain the output location for file loggers
    mainLoggerName : str
        name of the main info logger
    auxLoggerNames : List[str]
        names of additional loggers

    Returns
    -------
        None
    """
    logger = logging.getLogger(mainLoggerName)
    logger.setLevel(logging.INFO)

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    formatter = logging.Formatter('%(asctime)s - %(message)s', '%d-%b-%y %H:%M:%S')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(config.outDir / "info.log", mode='w')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if auxLoggerNames:
        for auxName in auxLoggerNames:
            auxLogger = logging.getLogger(auxName)
            while auxLogger.hasHandlers():
                auxLogger.removeHandler(auxLogger.handlers[0])

            auxLogger.setLevel(logging.INFO)

            fileHandler = logging.FileHandler(config.outDir / "{}.log".format(auxName), mode='w')
            fileHandler.setLevel(logging.INFO)
            auxLogger.addHandler(fileHandler)


class PadToSize(object):
    """
    Custom transformation that maintains the words original aspect ratio by scaling it to the given height and padding
    it to achieve the desired width.
    """

    def __init__(self, height, width, padwith=1):
        self.width = width
        self.height = height
        self.padwith = padwith

    def __call__(self, img):
        oldWidth, oldHeight = img.size
        if oldWidth != self.width or oldHeight != self.height:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            if intermediateWidth > self.width:
                intermediateWidth = self.width
            resized = img.resize((intermediateWidth, self.height), resample=Image.BICUBIC)
            preprocessed = Image.new('L', (self.width, self.height), self.padwith)
            preprocessed.paste(resized)
            return preprocessed
        else:
            return img

    @classmethod
    def invert(cls, image: np.ndarray, targetShape: Tuple[int, int]) -> np.ndarray:
        # resize so that the height matches, then cut off at width ...
        originalHeight, originalWidth = image.shape
        scaleFactor = targetShape[0] / originalHeight
        resized = rescale(image, scaleFactor)
        return resized[:, :targetShape[1]]

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


def composeTransformations(config: configuration.Configuration) -> transforms.Compose:
    """
    Composes various transformations based on the given experiment configuration. :class:`ToTensor` is always the final
    transformation.

    Parameters
    ----------
    config : Configuration
        experiment configuration

    Returns
    -------
    Compose
        the composed transformations
    """
    transformation = []

    if config.padScale:
        transformation.append(PadToSize(config.padHeight, config.padWidth, 255))
    transformation.extend([transforms.Resize((config.imageHeight, config.imageWidth)),
                           transforms.Grayscale(num_output_channels=1)])
    if config.invertImages:
        transformation.append(ImageOps.invert)
    transformation.append(transforms.ToTensor())

    return transforms.Compose(transformation)


def getModel(config: configuration.Configuration) -> torch.nn.Module:
    """
    Initialises the model based on the provided configuration.

    Parameters
    ----------
    config : Configuration
        experiment configuration

    Returns
    -------
    torch.nn.Module
        the paired image to image translation model

    """
    if config.modelName == ModelName.DENSE:
        return DenseGenerator(1, 1, n_blocks=config.blockCount)
    elif config.modelName == ModelName.SHALLOW:
        return Shallow(1, 1, )
    elif config.modelName == ModelName.TIRAMISU:
        model = Tiramisu(1, 1, structure=(
            config.down,  # Down blocks
            config.bottleneck,  # bottleneck layers
            config.up,  # Up blocks
        ), checkpoint=False)

        model.initialize_kernels(torch.nn.init.kaiming_uniform_, conv=True)
        return model
    else:
        return SimpleCNN()
