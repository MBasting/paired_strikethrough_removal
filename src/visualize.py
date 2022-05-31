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
                RMSE_dict[joined] = [[[] for j in range(7)] for i in range(6)]
                # For each stroketype there is an array holding the results for all models

                F1_dict[joined] = {k.name: [[] for i in range(7)] for k in StrikeThroughType}

            for strokeType, strokeRes in result_dataset["RMSE"].items():
                if strokeType != "all":
                    ind = StrikeThroughType.value_for_name(strokeType)
                    RMSE_dict[joined][index][ind].append(strokeRes)
                # for strokeType, strokeRes in result_dataset["F1"].items():
                #     F1_dict[joined][strokeType][index].append(strokeRes)
    ticks = [stroke.name for stroke in StrikeThroughType]
    colors = ["#7fc97f", "#beaed4", "#fdc086", "#bf5b17", "#386cb0", "#f0027f", "#ffff99"]
    locs = np.arange(-1.5, 1.6, 3/len(models))
    # Creat a plot for each training, test dataset combination
    for ttkey, train_test_data_comb in RMSE_dict.items():
        plt.figure(figsize=(10, 10))
        # Number of rows
        # Each row of train_test_dataCOMB are the results of scores for each stroketype
        for i, model_data in enumerate(train_test_data_comb):
            tmp = plt.boxplot(model_data, positions=np.array(range(len(model_data))) * 3 + locs[i], sym='', widths=0.3)
            set_box_color(tmp, colors[i])
            plt.plot([], c=colors[i], label=models[i])
        plt.legend(loc='upper left')
        plt.xticks(range(0, len(ticks) * 3, 3), ticks)
        plt.xlim(-2, len(ticks) * 3)
        plt.title(ttkey)
        plt.tight_layout()
        plt.savefig("figures/" + ttkey)

    # bp1 =
    # bp2 = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
    # bp3 = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
    # bp4 = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
    # bp5 = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
    # bp6 = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
    # bp7 = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
    #
    # set_box_color(bpl, '#a6cee3')  # colors are from http://colorbrewer2.org/
    # set_box_color(bpr, '#1f78b4')
    # set_box_color(bpr, '#1f78b4')
    # set_box_color(bpr, '#1f78b4')
    # set_box_color(bpr, '#1f78b4')
    # set_box_color(bpr, '#1f78b4')

    # print(RMSE_dict)
    # print(F1_dict)
    print(models)

if __name__ == '__main__':
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-result", required=True, help="path to file with results in json format")
    args = cmdParser.parse_args()
    visualize_plot(args.result)
