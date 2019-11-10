# TODO
# Choosing platform and device
# Timing

import cv2
from src.segmentation_algorithms import SegmentationAlgorithms


def main():
    img = cv2.imread('images/in.png', cv2.IMREAD_GRAYSCALE)

    algorithms = SegmentationAlgorithms()
    algorithms.iterative.run(img)
    print(algorithms.iterative.out_labels)
    
    algorithms.iterative.show_result()


if __name__ == '__main__':
    main()
