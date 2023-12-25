import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    skittles = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height = skittles.shape[0]
    width = skittles.shape[1]
    if height > width:
        skittles = cv2.resize(skittles, (720, 1280))
    elif width > height:
        skittles = cv2.resize(skittles, (1280, 720))

    H_low = {'green': 31, 'purple': 72, 'yellow': 20, 'red': 168}
    H_high = {'green': 60, 'purple': 168, 'yellow': 33, 'red': 180}
    S_low = {'green': 78, 'purple': 76, 'yellow': 123, 'red': 71}
    S_high = {'green': 255, 'purple': 255, 'yellow': 255, 'red': 255}
    V_low = {'green': 0, 'purple': 33, 'yellow': 92, 'red': 104}
    V_high = {'green': 255, 'purple': 142, 'yellow': 255, 'red': 255}

    skittles_hsv = cv2.cvtColor(skittles, cv2.COLOR_BGR2HSV)

    median_blur_value = {'green': 5, 'purple': 5, 'yellow': 5, 'red': 11}

    erode_matrix = {'green': np.ones((5, 5), "uint8"), 'purple': np.ones((3, 3), "uint8"),
                    'yellow': np.ones((7, 7), "uint8"), 'red': np.ones((5, 5), "uint8")}

    dilate_matrix = {'green': np.ones((5, 5), "uint8"), 'purple': np.ones((9, 9), "uint8"),
                     'yellow': np.ones((5, 5), "uint8"), 'red': np.ones((9, 9), "uint8")}

    ## COLOUR THRESHOLDS ##
    green = 0
    green_threshold = cv2.inRange(skittles_hsv, (H_low['green'], S_low['green'], V_low['green']),
                                  (H_high['green'], S_high['green'], V_high['green']))
    green_threshold = cv2.erode(green_threshold, erode_matrix['green'])
    green_threshold = cv2.medianBlur(green_threshold, median_blur_value['green'])
    green_contours, hierarchy = cv2.findContours(green_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for n in range(len(green_contours)):
        cv2.drawContours(skittles, green_contours, -1, (0, 255, 0), 1)
        green += 1

    purple = 0
    purple_threshold = cv2.inRange(skittles_hsv, (H_low['purple'], S_low['purple'], V_low['purple']),
                                  (H_high['purple'], S_high['purple'], V_high['purple']))
    purple_threshold = cv2.erode(purple_threshold, erode_matrix['purple'])
    purple_threshold = cv2.dilate(purple_threshold, dilate_matrix['purple'])
    purple_threshold = cv2.medianBlur(purple_threshold, median_blur_value['purple'])
    purple_contours, hierarchy = cv2.findContours(purple_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for n in range(len(purple_contours)):
        cv2.drawContours(skittles, purple_contours, -1, (238, 130, 238), 1)
        purple += 1

    yellow = 0
    yellow_threshold = cv2.inRange(skittles_hsv, (H_low['yellow'], S_low['yellow'], V_low['yellow']),
                                  (H_high['yellow'], S_high['yellow'], V_high['yellow']))
    yellow_threshold = cv2.erode(yellow_threshold, erode_matrix['yellow'])
    yellow_threshold = cv2.medianBlur(yellow_threshold, median_blur_value['yellow'])
    yellow_contours, hierarchy = cv2.findContours(yellow_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for n in range(len(yellow_contours)):
        cv2.drawContours(skittles, yellow_contours, -1, (0, 255, 255), 1)
        yellow += 1

    red = 0
    red_threshold = cv2.inRange(skittles_hsv, (H_low['red'], S_low['red'], V_low['red']),
                                  (H_high['red'], S_high['red'], V_high['red']))
    red_threshold = cv2.erode(red_threshold, erode_matrix['red'])
    red_threshold = cv2.dilate(red_threshold, erode_matrix['red'])
    red_threshold = cv2.medianBlur(red_threshold, median_blur_value['red'])
    red_contours, hierarchy = cv2.findContours(red_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for n in range(len(red_contours)):
        cv2.drawContours(skittles, red_contours, -1, (0, 0, 255), 1)
        red += 1

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
