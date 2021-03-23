from ..utils.constants import *
from ..utils.vector3 import vec3, rgb, extract
from functools import reduce as reduce
from ..ray import Ray, get_raycolor
from .. import lights
import numpy as np
from . import Material
from ..textures import *


class Emissive(Material):
    def __init__(self, color, **kwargs):

        if isinstance(color, vec3):
            self.texture_color = solid_color(color)
        elif isinstance(color, texture):
            self.texture_color = color

        super().__init__(**kwargs)

    def get_color(self, scene, ray, hit):
        n_rays = ray.p_z.shape[0]
        diff_color = self.texture_color.get_color(hit).repeat(n_rays)
        ray.color = diff_color
        return ray, np.array([])
