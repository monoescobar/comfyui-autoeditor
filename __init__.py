import sys


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

from .auto_editor import NODE_CLASS_MAPPINGS as AE_NODES, NODE_DISPLAY_NAME_MAPPINGS as AE_NAMES
from .audio_mixer import NODE_CLASS_MAPPINGS as AM_NODES, NODE_DISPLAY_NAME_MAPPINGS as AM_NAMES
from .lyrics_overlay import NODE_CLASS_MAPPINGS as LO_NODES, NODE_DISPLAY_NAME_MAPPINGS as LO_NAMES

NODE_CLASS_MAPPINGS = {**AE_NODES, **AM_NODES, **LO_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**AE_NAMES, **AM_NAMES, **LO_NAMES}

WEB_DIRECTORY = "./web"
