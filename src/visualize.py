import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from src.configuration import StrikeThroughType


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def visualize_plot(result_path):
    result_data = json.load(open(result_path))
    RMSE_dict = {}
    F1_dict = {}
    models = list()
    for ind_result, result_ind_value in result_data.items():
        if "ORIGINAL" in ind_result:
            split = ind_result.split("_")
            model = "CYCLE_GAN_ORIGINAL"
        else:
            split = ind_result.split("_")
            model = split[1]
        if model not in models:
            index = len(models)
            models.append(model)
        else:
            index = models.index(model)
        train_dataset = '_'.join(split[len(split) - 2:])

        for dataset_name, result_dataset in result_ind_value.items():
            test_dataset = dataset_name
            joined = '-'.join([train_dataset, test_dataset])

            if joined not in RMSE_dict:
                # For each stroketype there is an array holding the results for all models
                RMSE_dict[joined] = [[[] for j in range(7)] for i in range(6)]
                F1_dict[joined] = [[[] for j in range(7)] for i in range(6)]

            for strokeType, strokeRes in result_dataset["RMSE"].items():
                if strokeType != "all":
                    ind = StrikeThroughType.value_for_name(strokeType)
                    RMSE_dict[joined][index][ind].append(strokeRes)
            for strokeType, strokeRes in result_dataset["F1"].items():
                if strokeType != "all":
                    ind = StrikeThroughType.value_for_name(strokeType)
                    F1_dict[joined][index][ind].append(strokeRes)

    ticks = [stroke.name for stroke in StrikeThroughType]
    colors = ["#7fc97f", "#beaed4", "#fdc086", "#bf5b17", "#386cb0", "#f0027f", "#ffff99"]
    locs = np.arange(-1.5, 1.6, 3/len(models))

    for error_name, error_dict in zip(["RMSE", "F1"], [RMSE_dict, F1_dict]):
        # Creat a plot for each training, test dataset combination
        for ttkey, train_test_data_comb in error_dict.items():
            plt.figure(figsize=(10, 10))
            # Number of rows
            # Each row of train_test_dataCOMB are the results of scores for each stroketype
            for i, model_data in enumerate(train_test_data_comb):
                tmp = plt.boxplot(model_data, positions=np.array(range(len(model_data))) * 3 + locs[i], sym='', widths=0.3)
                set_box_color(tmp, colors[i])
                plt.plot([], c=colors[i], label=models[i])
            if error_name == "RMSE":
                plt.legend(loc='upper left')
            else:
                plt.legend(loc='upper right')
            plt.xticks(range(0, len(ticks) * 3, 3), ticks)
            plt.xlim(-2, len(ticks) * 3)
            plt.title(ttkey)
            plt.tight_layout()
            plt.savefig("figures/{}_{}".format(error_name, ttkey))

if __name__ == '__main__':
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-result", required=True, help="path to file with results in json format")
    args = cmdParser.parse_args()
    visualize_plot(args.result)
