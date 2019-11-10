import numpy as np
import pyopencl as cl
import sys

from src.segmentation import Segmentation


class IterativeSegmentation(Segmentation):
    """
    Implements embarrassingly parallel (EP) iterative segmentation
    algorithm.
    """

    def __init__(self, context, queue):
        super().__init__(context, queue)

    def _create_image_buff(self, image):
        """Create an image buffer which hold the image for OpenCL."""
        return cl.Image(
            self.context, cl.mem_flags.READ_ONLY,
            cl.ImageFormat(
                cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8
            ),
            shape=image.shape
        )

    def _create_labels_buff(self, image):
        """Create an image buffer which hold labels for OpenCL."""
        return cl.Image(
            self.context, cl.mem_flags.READ_WRITE,
            cl.ImageFormat(
                cl.channel_order.R, cl.channel_type.UNSIGNED_INT32
            ),
            shape=image.shape
        )

    def _create_compare_result_buff(self, image, comp_res):
        """
        Create a buffer which hold the result of comparing two matrices.
        """
        return cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=comp_res
        )

    def _main_loop(
            self, image, image_buff, labels_in_buff,
            labels_out_buff, comp_res_buff, init_labels_kernel,
            min_label_kernel, comp_matrices_kernel, labels, comp_res):
        """Main loop of iterative algorithm."""
        # Copy image to GPU, init labels
        cl.enqueue_copy(
            self.queue, image_buff, image, origin=(0, 0),
            region=image.shape, is_blocking=False
        )
        init_labels_task = cl.enqueue_nd_range_kernel(
            self.queue, init_labels_kernel, image.shape, None
        )

        # Update labels until stability
        it_counter = 1
        while True:
            find_min = cl.enqueue_nd_range_kernel(
                self.queue, min_label_kernel, image.shape, None,
                wait_for=[init_labels_task]
            )
            comp_mat = cl.enqueue_nd_range_kernel(
                self.queue, comp_matrices_kernel, image.shape, None,
                wait_for=[find_min]
            )

            # Check if there was a change, if not, break
            cl.enqueue_copy(
                self.queue, comp_res, comp_res_buff, wait_for=[comp_mat],
                is_blocking=True
            )
            if np.sum(comp_res) == 0:
                break

            # Swap input with output
            cl.enqueue_copy(
                self.queue, labels_in_buff, labels_out_buff, src_origin=(0, 0),
                dest_origin=(0, 0), region=image.shape
            )

            it_counter += 1

        # Copy the data from buffer to host
        cl.enqueue_copy(
            self.queue, labels, labels_out_buff, origin=(0, 0),
            region=image.shape, is_blocking=True
        )

        print('Took {0} iterations.'.format(it_counter))

    def run(self, image):
        labels = np.empty(image.shape, dtype=np.uint32)
        comp_res = np.zeros(image.shape[0] * image.shape[1]).astype(np.float32)

        # Create buffers for OpenCL
        image_buff = self._create_image_buff(image)
        labels_in_buff = self._create_labels_buff(image)
        labels_out_buff = self._create_labels_buff(image)
        comp_res_buff = self._create_compare_result_buff(image, comp_res)

        # Load kernels
        program = self._load_program('kernel.cl')
        init_labels_kernel = self._setup_kernel(
            program, 'init_labels', image_buff, labels_in_buff
        )
        min_label_kernel = self._setup_kernel(
            program, 'neigh_min_label', labels_in_buff, labels_out_buff
        )
        comp_matrices_kernel = self._setup_kernel(
            program, 'compare_matrices', labels_in_buff, labels_out_buff,
            comp_res_buff
        )

        # Run main loop of algorithm
        self._main_loop(
            image, image_buff, labels_in_buff, labels_out_buff, comp_res_buff,
            init_labels_kernel, min_label_kernel, comp_matrices_kernel, labels,
            comp_res
        )

        self.out_labels = labels
