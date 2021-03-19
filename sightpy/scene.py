from PIL import Image
import numpy as np
import time
import copy
from multiprocessing import Pool, cpu_count
from .utils import colour_functions as cf
from .camera import Camera
from .utils.constants import *
from .utils.vector3 import vec3, rgb
from .ray import Ray, get_raycolor, get_distances
from . import lights
from .backgrounds.skybox import SkyBox
from .backgrounds.panorama import Panorama


def get_raycolor_tuple(x):
    return get_raycolor(*x)


class Scene:
    def __init__(self, ambient_color=rgb(0.01, 0.01, 0.01), n=vec3(1.0, 1.0, 1.0)):
        # n = index of refraction (by default index of refraction of air n = 1.)

        self.scene_primitives = []
        self.collider_list = []
        self.shadowed_collider_list = []
        self.Light_list = []
        self.importance_sampled_list = []
        self.ambient_color = ambient_color
        self.n = n
        self.importance_sampled_list = []

    def add_Camera(self, look_from, look_at, **kwargs):
        self.camera = Camera(look_from, look_at, **kwargs)

    def add_PointLight(self, pos, color):
        self.Light_list += [lights.PointLight(pos, color)]

    def add_DirectionalLight(self, Ldir, color):
        self.Light_list += [lights.DirectionalLight(Ldir.normalize(), color)]

    def add(self, primitive, importance_sampled=False):
        self.scene_primitives += [primitive]
        self.collider_list += primitive.collider_list

        if importance_sampled == True:
            self.importance_sampled_list += [primitive]

        if primitive.shadow == True:
            self.shadowed_collider_list += primitive.collider_list

    def add_Background(self, img, light_intensity=0.0, blur=0.0, spherical=False):

        primitive = None
        if spherical == False:
            primitive = SkyBox(img, light_intensity=light_intensity, blur=blur)
        else:
            primitive = Panorama(img, light_intensity=light_intensity, blur=blur)

        self.scene_primitives += [primitive]
        self.collider_list += primitive.collider_list

    def render(self, samples_per_pixel, progress_bar=False):

        print("Rendering...")

        t0 = time.time()
        color_RGBlinear = rgb(0.0, 0.0, 0.0)

        all_rays = [
            (self.camera.get_ray(self.n), copy.deepcopy(self))
            for i in range(samples_per_pixel)
        ]
        if progress_bar == True:
            try:
                import progressbar
            except ModuleNotFoundError:
                print("progressbar module is required. \nRun: pip install progressbar")

            bar = progressbar.ProgressBar(maxval=samples_per_pixel)
            n_proc = cpu_count()

            with Pool(processes=n_proc) as pool:
                bar.start()
                for i, color in enumerate(
                    pool.imap_unordered(get_raycolor_tuple, all_rays)
                ):
                    color_RGBlinear += color
                    bar.update(i)
                bar.finish()

        else:
            with Pool(processes=n_proc) as pool:
                for color in pool.imap_unordered(get_raycolor_tuple, all_rays):
                    color_RGBlinear += color

        # average samples per pixel (antialiasing)
        color_RGBlinear = color_RGBlinear / samples_per_pixel
        # gamma correction
        color = cf.sRGB_linear_to_sRGB(color_RGBlinear.to_array())

        print("Render Took", time.time() - t0)

        img_RGB = []
        for c in color:
            # average ray colors that fall in the same pixel. (antialiasing)
            img_RGB += [
                Image.fromarray(
                    (
                        255
                        * np.clip(c, 0, 1).reshape(
                            (self.camera.screen_height, self.camera.screen_width)
                        )
                    ).astype(np.uint8),
                    "L",
                )
            ]

        return Image.merge("RGB", img_RGB)

    def get_distances(
        self,
    ):  # Used for debugging ray-primitive collisions. Return a grey map of objects distances.

        print("Rendering...")
        t0 = time.time()
        color_RGBlinear = get_distances(self.camera.get_ray(self.n), scene=self)
        # gamma correction
        color = color_RGBlinear.to_array()

        print("Render Took", time.time() - t0)

        img_RGB = [
            Image.fromarray(
                (
                    255
                    * np.clip(c, 0, 1).reshape(
                        (self.camera.screen_height, self.camera.screen_width)
                    )
                ).astype(np.uint8),
                "L",
            )
            for c in color
        ]
        return Image.merge("RGB", img_RGB)
