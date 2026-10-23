"""Original RGB <-> official VGGT pad-mode continuous pixel coordinates."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VGGTPointCoordinates:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    network_size: int = 518
    crop_x: int = 0
    crop_y: int = 0

    @classmethod
    def from_shape(cls, shape, size=518):
        height, width = map(int, shape[:2])
        if min(height, width) <= 0 or size % 14:
            raise ValueError('Positive image size and patch-aligned network size required')
        if width >= height:
            rw, rh = size, round(height * (size / width) / 14) * 14
        else:
            rh, rw = size, round(width * (size / height) / 14) * 14
        if min(rw, rh) <= 0:
            raise ValueError('Aspect ratio collapses an official rounded resize dimension')
        return cls(width, height, rw, rh, (size-rw)//2, (size-rh)//2, size)

    @property
    def scale(self):
        return np.array([self.resized_width/self.original_width,
                         self.resized_height/self.original_height], np.float64)

    def to_network(self, xy):
        points = np.asarray(xy, np.float64)
        if points.shape[-1:] != (2,):
            raise ValueError('Expected xy pixel coordinates')
        # PIL resize operates on pixel centers: (x+.5)*scale-.5.
        return (points+.5) * self.scale - .5 + [self.pad_left-self.crop_x, self.pad_top-self.crop_y]

    def to_original(self, xy):
        points = np.asarray(xy, np.float64)
        if points.shape[-1:] != (2,):
            raise ValueError('Expected xy pixel coordinates')
        return (points - [self.pad_left-self.crop_x, self.pad_top-self.crop_y] + .5) / self.scale - .5

    def valid_original(self, xy):
        xy = np.asarray(xy)
        return (np.isfinite(xy).all(-1) & (xy[..., 0] >= -1e-7) & (xy[..., 1] >= -1e-7)
                & (xy[..., 0] < self.original_width) & (xy[..., 1] < self.original_height))

    def valid_network(self, xy):
        return self.valid_original(self.to_original(xy))

    def preprocess(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.shape != (self.original_height,self.original_width,3) or rgb.dtype != np.uint8:
            raise ValueError('Expected original uint8 RGB image')
        resized = np.asarray(Image.fromarray(rgb).resize((self.resized_width,self.resized_height),
                                                        Image.Resampling.BICUBIC))
        out = np.full((self.network_size,self.network_size,3),255,np.uint8)
        out[self.pad_top:self.pad_top+self.resized_height,
            self.pad_left:self.pad_left+self.resized_width] = resized
        return np.ascontiguousarray(out.transpose(2,0,1),dtype=np.float32)/255.0

    def manifest(self):
        return {**asdict(self),'scale_xy':self.scale.tolist(),
                'mode':'official pad','point_convention':'pixel-center xy; PIL half-pixel resize plus padding',
                'padding_rgb':[255,255,255]}
