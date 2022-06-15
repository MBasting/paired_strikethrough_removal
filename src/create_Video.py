import argparse

import cv2
import numpy as np
from pathlib import Path

def visualize_video(folder):
    # choose codec according to format needed
    folder = Path('exp1/ORIGINAL_1654415455_68419_IAMsynth_full_4')
    index = 6

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_cleanConcat = None
    video_struckConcat = None
    for i in range(2, 30, 2):
        img_cleanConcat = Path(str(folder) + f"/{i}/cleanConcat_e{i}_b{index}.png")
        img_cleanConcat = cv2.imread(str(img_cleanConcat.absolute()))
        if "ORIGINAL" in folder.name:
            img_struckConcat = Path(str(folder) + f"/{i}/struckConcat_e{i}_b{index}.png")
            img_struckConcat = cv2.imread(str(img_struckConcat.absolute()))

        if i == 2:
            height1 = img_cleanConcat.shape[0]
            width1 = img_cleanConcat.shape[1]
            video_cleanConcat = cv2.VideoWriter(f'videos/video_{folder.name}_Image_{index}_StrucktoCleanConcat.avi', fourcc, 3,
                                                (width1, height1))
            if "ORIGINAL" in folder.name:
                height2 = img_struckConcat.shape[0]
                width2 = img_struckConcat.shape[1]
                video_struckConcat = cv2.VideoWriter(f'videos/video_{folder.name}_Image_{index}_CleantoStruckConcat.avi', fourcc, 3,
                                                     (width2, height2))

        video_cleanConcat.write(img_cleanConcat)
        if "ORIGINAL" in folder.name:
            video_struckConcat.write(img_struckConcat)

    cv2.destroyAllWindows()
    video_cleanConcat.release()
    video_struckConcat.release()
    #     for j in range(0, 5):
#
#
if __name__ == '__main__':
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-folder", required=True, help="path to config file", default=None)
    args = cmdParser.parse_args()
    visualize_video(args.folder)