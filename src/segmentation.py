import pyopencl as cl
import numpy as np
import random
import cv2


class Segmentation():
    """Abstract class for segmentation algorithms."""

    def __init__(self, context, queue):
        self.context = context
        self.queue = queue
        self.out_labels = None

    def _load_program(self, kernel):
        """Load and compile OpenCL program."""
        return cl.Program(
            self.context, open('kernels/{0}'.format(kernel)).read()
        ).build()

    def _setup_kernel(self, program, kernel_name, *argv):
        """Get kernel from OpenCL program and set arguments."""
        kernel = cl.Kernel(program, kernel_name)
        for idx, value in enumerate(argv):
            kernel.set_arg(idx, value)

        return kernel

    def _random_color(self):
        """Create random color in RGB format."""
        levels = range(0, 256)
        return tuple(random.choice(levels) for _ in range(3))

    def _create_color_map(self):
        """Create dict to map every unique label to color."""
        unique_labels = np.unique(self.out_labels)
        color_map = {}
        for unique_label in unique_labels:
            color_map[unique_label] = self._random_color()

        return color_map

    def _create_segmented_image(self, color_map):
        """Create RGB image, where every label has different color."""
        height, width = self.out_labels.shape
        segmented_image = np.zeros((height, width, 3), np.uint8)

        for label, color in color_map.items():
            segmented_image[self.out_labels == label] = color

        return segmented_image

    def show_result(self):
        """Show segmented image."""
        if self.out_labels is None:
            print('You have to call run() first.')
            return

        color_map = self._create_color_map()
        segmented_image = self._create_segmented_image(color_map)

        cv2.imshow('Segmented image', segmented_image)
        cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

    def run(self, image):
        raise NotImplementedError
