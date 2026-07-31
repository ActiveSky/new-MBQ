"""Offline collection shim for lmms-eval's optional hf_transfer import.

The collector never downloads files when --local-files-only is set, so the
native transfer extension is not needed for that narrow execution path.
"""

