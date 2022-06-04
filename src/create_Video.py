import cv2
import numpy as np
from pathlib import Path

# choose codec according to format needed
folder = Path('tmp/ORIGINAL/ORIGINAL_1654011739_87359_IAMsynth_full')
index = 6

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_cleanConcat = None
video_struckConcat = None
for i in range(5, 30, 5):
    img_cleanConcat = Path(str(folder) + f"/{i}/cleanConcat_e{i}_b{index}.png")
    img_cleanConcat = cv2.imread(str(img_cleanConcat.absolute()))
    if "ORIGINAL" in folder.name:
        img_struckConcat = Path(str(folder) + f"/{i}/struckConcat_e{i}_b{index}.png")
        img_struckConcat = cv2.imread(str(img_struckConcat.absolute()))

    if i == 5:
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
