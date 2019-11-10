import pyopencl as cl
from src.iterative_segmentation import IterativeSegmentation


class SegmentationAlgorithms():
    """Algorithms for image segmentation implemented on GPU."""

    def __init__(self):
        self._setup_opencl()
        self._show_device()

        self.iterative = IterativeSegmentation(self.context, self.queue)

    def _setup_opencl(self):
        """Select GPU device, create Context and Queue."""
        platforms = cl.get_platforms()
        platform = platforms[1]
        devices = platform.get_devices(cl.device_type.GPU)

        self.device = devices[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, self.device)

    def _show_device(self):
        print("Running on {0}.".format(self.device.name))
