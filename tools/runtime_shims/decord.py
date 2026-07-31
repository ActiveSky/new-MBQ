"""Image-only calibration shim for an optional video decoder dependency."""


class VideoReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "VideoReader is unavailable in the offline collection runtime. "
            "This collector is restricted to the COCO image calibration set."
        )


def cpu(index=0):
    return index
