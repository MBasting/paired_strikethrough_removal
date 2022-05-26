"""
Code to test a previously trained neural network regarding its performance of removing strikethrough from a word.
"""
import argparse
import configparser
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src import metrics
from src.configuration import Configuration, ModelName, DatasetChoice
from src.cycle_gan import ConfigurationGAN
from src.cycle_gan.test_gan import TestRunnerGAN
from src.dataset import TestDataset
from src.utils import composeTransformations, PadToSize, initLoggers, getModel

import warnings

warnings.simplefilter("ignore", UserWarning)

INFO_LOGGER_NAME = "st_removal"
RESULTS_LOGGER_NAME = "results"


class TestRunner:
    """
    Utility class that wraps the initialisation and testing of a neural network.
    """

    def __init__(self, configuration: Configuration, testImageDir, saveCleanedImages: bool = True,
                 model_name: str = "best_fmeasure.pth"):
        self.logger = logging.getLogger(INFO_LOGGER_NAME)
        self.resultsLogger = logging.getLogger(RESULTS_LOGGER_NAME)
        self.config = configuration
        self.saveCleanedImages = saveCleanedImages

        transformations = composeTransformations(self.config)
        testDataset = TestDataset(testImageDir, transformations, strokeTypes=["all"])
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

        rmses = []
        fmeasures = []

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
                    fmeasures.append(f1)
                    rmses.append(rmse)
                    self.resultsLogger.info('%f,%f,%s,%s', rmse, f1[0], strokeType, imagePath)

        self.resultsLogger.info('%f,%f', np.mean(rmses), np.mean(fmeasures))


def evaluate_one_file(file, data, save, checkpoint, GAN=False):
    configPath = Path(file)
    if data is not None:
        dataPath = Path(data)
    saveCleanedImages = save
    modelName = checkpoint

    configParser = configparser.ConfigParser()
    configParser.read(configPath)

    section = "DEFAULT"
    sections = configParser.sections()
    if len(sections) == 1:
        section = sections[0]
    else:
        logging.getLogger("st_recognition").warning("Found %s than one named sections in config file. Using 'DEFAULT' "
                                                    "as fallback.", 'more' if len(sections) > 1 else 'fewer')
    print(file)
    parsedConfig = configParser[section]
    if not GAN:
        conf = Configuration(parsedConfig, test=True, fileSection=section)
        dataset_dir = conf.dataset_dir
    else:
        conf = ConfigurationGAN(parsedConfig, test=True, fileSection=section)
        dataset_dir = "datasets"

    if data is not None:
        testImageDirs = [dataPath]
    else:
        testImageDirs = [Path(dataset_dir + "/" + ds_choice.name + "/test") for ds_choice in DatasetChoice if
                             ds_choice.name != "Dracula_synth"]
    results = {}
    for testImageDir in testImageDirs:
        conf.testImageDir = testImageDir
        out = configPath.parent / "{}_{}".format(testImageDir.parent.name, testImageDir.name)
        out.mkdir(exist_ok=True)
        conf.outDir = out
        results[testImageDir.parent.name] = conf.outDir.name + "/results.log"
        initLoggers(conf, INFO_LOGGER_NAME, [RESULTS_LOGGER_NAME])
        logger = logging.getLogger(INFO_LOGGER_NAME)
        logger.info(conf.outDir)
        if GAN:
            runner = TestRunnerGAN(conf, testImageDir, saveCleanedImages=saveCleanedImages)
            runner.test()
        else:
            runner = TestRunner(conf, testImageDir, saveCleanedImages=saveCleanedImages)
            runner.test()
    return results


def evaluate_folder(folder, data, save, checkpoint, min_time=None):
    results = {}
    for child in folder.iterdir():
        if child.is_dir():
            # Found folder that contains CycleGAN results
            if len(child.name.split("_")) < 2:
                for grand_child in child.iterdir():
                    file = folder.name + "/" + child.name + "/" + grand_child.name + "/config.cfg"
                    results_files = evaluate_one_file(file, data, save, checkpoint, True)
                    temp = read_to_dict(results_files, path=folder.name + "/" + child.name + "/" + grand_child.name)
                    results[grand_child.name] = temp

            elif min_time is None or int(child.name.split("_")[-3]) > min_time:
                file = folder.name + "/" + child.name + "/config.cfg"
                results_files = evaluate_one_file(file, data, save, checkpoint)
                temp = read_to_dict(results_files, path=folder.name + "/" + child.name)
                results[child.name] = temp
    with open(f'results_{str(time.time())}.json', 'w') as fp:
        json.dump(results, fp)


def read_to_dict(results_files, path):
    temp = {}
    for res_name, res_file in results_files.items():
        temp2 = {}
        path_to_res = path + "/" + res_file
        stroke_type_rmse, stroke_type_f1, stroke_counts = read_log(path_to_res)
        temp2["RMSE"] = stroke_type_rmse
        temp2["F1"] = stroke_type_f1
        temp[res_name] = temp2
    return temp


def read_log(path_to_res):
    with open(path_to_res) as f:
        f = f.readlines()
    f = f[1:]
    stroke_type_rmse = {}
    stroke_type_f1 = {}
    stroke_counts = {}
    for lines in f:
        split = lines.split(",")
        if len(split) == 2:
            continue

        if stroke_type_rmse.__contains__(split[2]):
            stroke_type_rmse[split[2]] += float(split[0])
            stroke_type_f1[split[2]] += float(split[1])
            stroke_counts[split[2]] += 1
        else:
            stroke_type_rmse[split[2]] = float(split[0])
            stroke_type_f1[split[2]] = float(split[1])
            stroke_counts[split[2]] = 1
    total_rmse = 0
    total_f1 = 0

    for key, value in stroke_type_rmse.items():
        total_rmse += stroke_type_rmse[key]
        total_f1 += stroke_type_f1[key]
        stroke_type_rmse[key] /= stroke_counts[key]
        stroke_type_f1[key] /= stroke_counts[key]
    stroke_type_rmse["all"] = total_rmse / sum(stroke_counts.values())
    stroke_type_f1["all"] = total_f1 / sum(stroke_counts.values())
    return stroke_type_rmse, stroke_type_f1, stroke_counts


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-file", required=False, help="path to config file", default=None)
    cmdParser.add_argument("-folder", required=False, help="path to folder containing the training files", default=None)
    cmdParser.add_argument("-data", required=False,
                           help="path to data directory, if no path given all test dataset are run", default=None)
    cmdParser.add_argument("-save", required=False, help="saves cleaned images if given", default=False,
                           action='store_true')
    cmdParser.add_argument("-checkpoint", required=False,
                           help="checkpoint file name (incl. '.pth' extension) - Default: best_fmeasure.pth",
                           default="best_fmeasure.pth")
    args = cmdParser.parse_args()
    if args.file is None and args.folder is None:
        print("No file or Folder specified")
        exit()
    if args.folder is not None:
        folder = args.folder
        evaluate_folder(Path(folder), args.data, args.save, args.checkpoint)
    else:
        evaluate_one_file(args.file, args.data, args.save, args.checkpoint)
