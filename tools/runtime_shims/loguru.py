"""Small compatibility shim for the collector's offline runtime only."""

import logging


class _Logger:
    def __init__(self):
        self._logger = logging.getLogger("mbq-runtime-shim")

    def remove(self, *args, **kwargs):
        return None

    def add(self, *args, **kwargs):
        return 0

    def __getattr__(self, name):
        return getattr(self._logger, name)


logger = _Logger()
