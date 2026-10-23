"""Pillow compatibility helpers.

Pillow 9 introduced ``Image.Resampling`` while older installations expose
the same filter constants directly on ``Image``.  The experiment must run in
both environments without changing pixel-resampling semantics.
"""

from PIL import Image


RESAMPLING = getattr(Image, "Resampling", Image)

