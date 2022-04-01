"""
Code to test a previously trained neural network regarding its performance of removing strikethrough from a word.
"""
import argparse
import configparser
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src import metrics
from src.configuration import Configuration, ModelName
from src.dataset import TestDataset
from src.utils import composeTransformations, PadToSize, initLoggers, getModel

INFO_LOGGER_NAME = "st_removal"
RESULTS_LOGGER_NAME = "results"


class TestRunner:
    """
    Utility class that wraps the initialisation and testing of a neural network.
    """

    def __init__(self, configuration: Configuration, saveCleanedImages: bool = True,
                 model_name: str = "genStrikeToClean_best_fmeasure.pth"):
        self.logger = logging.getLogger(INFO_LOGGER_NAME)
        self.resultsLogger = logging.getLogger(RESULTS_LOGGER_NAME)
        self.config = configuration
        self.saveCleanedImages = saveCleanedImages

        transformations = composeTransformations(self.config)
        testDataset = TestDataset(self.config.testImageDir, transformations, strokeTypes=["all"])
        self.validationDataloader = DataLoader(testDataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

        self.model = getModel(self.config)

        modelBasePath = self.config.outDir.parent
        state_dict = torch.load(modelBasePath / model_name, map_location=torch.device(self.config.device))
        if "model_state_dict" in state_dict.keys():
            state_dict = state_dict['model_state_dict']

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.config.device)

        self.logger.info('Data dir: %s', str(self.config.testImageDir))
        self.logger.info('Validation dataset size: %d', len(testDataset))

    def test(self) -> None:
        self.model.eval()
        self.resultsLogger.info('rmse,f1,strike_type,image_id')

        if self.saveCleanedImages:
            imgDir = self.config.outDir / "images"
            imgDir.mkdir(exist_ok=True, parents=True)
        else:
            imgDir = self.config.outDir

        with torch.no_grad():
            for datapoints in self.validationDataloader:
                strokeTypes = datapoints['strokeType']
                struckImages = datapoints['struck'].to(self.config.device)
                imageSize = datapoints['image_size']
                cleanedImages = self.model(struckImages)

                if self.config.modelName == ModelName.TIRAMISU:
                    cleanedImages = torch.sigmoid(cleanedImages)

                groundTruthImages = datapoints['groundTruth'].cpu().numpy()

                for idx, imagePath in enumerate(datapoints['path']):
                    strokeType = strokeTypes[idx]
                    cleanedImage = PadToSize.invert(cleanedImages[idx].squeeze().cpu().numpy(),
                                                    (imageSize[0][idx], imageSize[1][idx]))
                    groundTruth = PadToSize.invert(groundTruthImages[idx].squeeze(),
                                                   (imageSize[0][idx], imageSize[1][idx]))

                    if self.saveCleanedImages:
                        Image.fromarray((1.0 - cleanedImage) * 255).convert("L").save(str(imgDir / 'cleaned_{}.png'.
                                                                                          format(imagePath)))

                    rmse = metrics.calculateRmse(groundTruth, cleanedImage)[0]

                    if self.config.invertImages:
                        f1 = metrics.calculateF1Score(255.0 - groundTruth * 255.0,
                                                      255.0 - cleanedImage * 255.0, binarise=True)[0]
                    else:
                        f1 = metrics.calculateF1Score(groundTruth * 255.0, cleanedImage * 255.0, binarise=True)[0]

                    self.resultsLogger.info('%f,%f,%s,%s', rmse, f1[0], strokeType, imagePath)


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-configfile", required=True, help="path to config file")
    cmdParser.add_argument("-data", required=True, help="path to data directory")
    cmdParser.add_argument("-save", required=False, help="saves cleaned images if given", default=False,
                           action='store_true')
    cmdParser.add_argument("-checkpoint", required=False, help="checkpoint file name (incl. '.pth' extension",
                           default="best_fmeasure.pth")
    args = cmdParser.parse_args()
    configPath = Path(args.configfile)
    dataPath = Path(args.data)
    saveCleanedImages = args.save
    modelName = args.checkpoint

    configParser = configparser.ConfigParser()
    configParser.read(configPath)

    section = "DEFAULT"

    sections = configParser.sections()
    if len(sections) == 1:
        section = sections[0]
    else:
        logging.getLogger("st_recognition").warning("Found %s than one named sections in config file. Using 'DEFAULT' "
                                                    "as fallback.", 'more' if len(sections) > 1 else 'fewer')

    parsedConfig = configParser[section]
    conf = Configuration(parsedConfig, test=True, fileSection=section)
    conf.testImageDir = dataPath

    out = configPath.parent / "{}_{}".format(dataPath.parent.name, dataPath.name)
    out.mkdir(exist_ok=True)
    conf.outDir = out

    initLoggers(conf, INFO_LOGGER_NAME, [RESULTS_LOGGER_NAME])
    logger = logging.getLogger(INFO_LOGGER_NAME)
    logger.info(conf.outDir)

    runner = TestRunner(conf, saveCleanedImages=saveCleanedImages, model_name=modelName)
    runner.test()
