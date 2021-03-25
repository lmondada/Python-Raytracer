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


def batch_rays(rays, batch_size):
    batches = []
    n_rays = len(rays)
    for ray_ind in range(0, n_rays, batch_size):
        batches.append(Ray.concatenate(rays[ray_ind : ray_ind + batch_size]))
    return batches


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

    def render(self, samples_per_pixel, progress_bar=False, batch_size=None):

        print("Rendering...")

        t0 = time.time()
        color_RGBlinear = rgb(0.0, 0.0, 0.0)

        all_rays = [self.camera.get_ray(self.n) for i in range(samples_per_pixel)]

        n_proc = cpu_count()
        rays_per_batch = len(self.camera.get_ray(self.n))
        batch_size = batch_size or np.ceil(samples_per_pixel / n_proc).astype(int)

        all_rays_batched = batch_rays(all_rays, batch_size)
        args = [(batch, copy.deepcopy(self)) for batch in all_rays_batched]
        # all_rays = [
        #     (self.camera.get_ray(self.n), copy.deepcopy(self))
        #     for i in range(samples_per_pixel)
        # ]

        def compute_cols(ray, copy_order, mining):
            # Aggregate the colours per pixel
            n_pixels = ray.p_z.shape[0] - np.array(copy_order).shape[0]
            full_copy_order = np.concatenate((
                np.array(range(n_pixels)),
                copy_order
            ))
            av_col_per_pixel = [
                ray.color.extract(full_copy_order == i).mean(axis=0)
                for i in range(n_pixels)
            ]
            combined_cols = av_col_per_pixel[0].append(tuple(av_col_per_pixel[1:]))

            # Aggregate the mining statistics
            mining["col_probs"] = np.concatenate((mining["col_probs"], ray.p_z))
            mining["col_probs_ref"] = np.concatenate((mining["col_probs_ref"], ray.p_z_ref))

            return combined_cols, mining

        def refine(dust):
            """Sort the gold out from the dust:
            Since we collected the data backwards, we need to reverse it to compute the actual statistics.
            Variables:
             - Z: ray state for each ray in the image that we are tracking
             - Z_i: ray states after i bounces (counting from the light source)
             - z, z_i: state for a single ray (after i bounces)
             - x: final image

            We want to collect:
             - Joint Score: t(x,Z|θ) = log p(x,Z|θ) = SUM[ log p(Z_i|θ,Z_{<=i}) ] + log p(x|θ,Z), for each θ_i.
             - Joint Likelihood Ratio: r(x,Z|θ_0, θ_1) = p(x,Z|θ_0)/p(x,Z|θ_1)

            We have accumulated:
             - p(z_i <- z_i-1|θ) transition probability for each i for each ray.

            We know:
             - p(x|θ,Z) = 1 (since x is just an average of the final ray colours/positions that land in the detector).
             - > p(x,Z|θ) = p(Z|θ) by law of conditional probability.
             - > r(x,Z|θ_0, θ_1) = p(Z|θ_0)/p(Z|θ_1) = exp[js(θ_0) - js(θ_1)]

            for each ray:
             - p(z_i|z<i,θ) = p(z_i <- z_i-1|θ) ie prob that we transition to z_i from z_i-1
             - > log p(z|θ) = SUM_i log p(z_i -> z_i+1|θ) transition/bounce probability

            combining the rays:
             - p(Z_i|θ,z_<i) = PROD_z p(z_i|θ,z<i) since each ray is independent of the others.
             - > log p(Z_i|θ,Z_<i) = SUM_z log p(z_i|θ,z<i) = SUM_z SUM_j<i log p(z_j <- z_j-1|θ)

            So mining the gold is just a case of summing over the log probs of complete paths.
            We can ignore incomplete paths (paths that didn't make it to the detector in time).
            This is ok to do, since they contribute nothing to the final image anyway (they add 0).
            """
            # first, we need to filter out runs that didn't hit the light (prob -1)
            clean_pz = dust["col_probs"][dust["col_probs"] >= 0]
            clean_pz_ref = dust["col_probs_ref"][dust["col_probs_ref"] >= 0]

            js_0 = sum(np.log(clean_pz))
            js_1 = sum(np.log(clean_pz_ref))
            jlr = np.exp(js_0 - js_1)

            print("Joint score:", js_0)
            print("Reference joint score:", js_1)
            print("Joint likelihood ratio:", jlr)

            return js_0, js_1, jlr

        # Keep track of all the gold dust we are mining
        mined_dust = {
            "col_probs": [],
            "col_probs_ref": [],
        }

        if progress_bar == True:
            try:
                import progressbar
            except ModuleNotFoundError:
                print("progressbar module is required. \nRun: pip install progressbar")

            bar = progressbar.ProgressBar(maxval=samples_per_pixel)

            with Pool(processes=n_proc) as pool:
                bar.start()
                for i, (ray, copy_order) in enumerate(
                        pool.imap_unordered(get_raycolor_tuple, args)
                ):
                    for batch in range(batch_size):
                        beg, end = batch * rays_per_batch, (batch + 1) * rays_per_batch
                        color, mined_dust = compute_cols(ray[beg:end], copy_order, mined_dust)
                        color_RGBlinear += color[beg:end]
                    bar.update(i)
                bar.finish()

        else:
            with Pool(processes=n_proc) as pool:
                for i, (ray, copy_order) in enumerate(
                    pool.imap_unordered(get_raycolor_tuple, args)
                ):
                    color, mined_dust = compute_cols(ray, copy_order, mined_dust)
                    for batch in range(batch_size):
                        beg, end = batch * rays_per_batch, (batch + 1) * rays_per_batch
                        color_RGBlinear += color[beg:end]

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

        gold_bars = refine(mined_dust)
        return Image.merge("RGB", img_RGB), gold_bars

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
