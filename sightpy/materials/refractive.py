from ..utils.constants import *
from ..utils.vector3 import vec3, rgb, extract
from functools import reduce as reduce
from ..ray import Ray, get_raycolor
from .. import lights
import numpy as np
from . import Material


class Refractive(Material):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)

        self.n = n  # index of refraction

        # Instead of defining a index of refraction (n) for each wavelenght (computationally expensive) we aproximate defining the index of refraction
        # using a vec3 for red = 630 nm, green 555 nm, blue 475 nm, the most sensitive wavelenghts of human eye.

        # Index a refraction is a complex number.
        # The real part is involved in how much light is reflected and model refraction direction via Snell Law.
        # The imaginary part of n is involved in how much light is reflected and absorbed. For non-transparent materials like metals is usually between (0.1j,3j)
        # and for transparent materials like glass is  usually between (0.j , 1e-7j)

    def get_color(self, scene, ray, hit):

        hit.point = ray.origin + ray.dir * hit.distance  # intersection point
        N = hit.material.get_Normal(hit)  # normal

        color = rgb(0.0, 0.0, 0.0)

        V = ray.dir * -1.0  # direction to ray origin
        nudged = hit.point + N * 0.000001  # M nudged to avoid itself
        # compute reflection and refraction
        # a paper explaining formulas used:
        # https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
        # reallistic refraction is expensive. (requires exponential complexity because each ray is divided in two)

        if ray.depth < hit.surface.max_ray_depth:

            """
            if hit_orientation== UPWARDS:
               #ray enter in the material
            if hit_orientation== UPDOWN:
               #ray get out of the material   
            """
            n1 = ray.n
            n2 = vec3.where(hit.orientation == UPWARDS, self.n, scene.n)

            n1_div_n2 = vec3.real(n1) / vec3.real(n2)
            cosθi = V.dot(N)
            sin2θt = (n1_div_n2) ** 2 * (1.0 - cosθi ** 2)

            # compute complete fresnel term
            cosθt = vec3.sqrt(1.0 - (n1 / n2) ** 2 * (1.0 - cosθi ** 2))
            r_per = (n1 * cosθi - n2 * cosθt) / (n1 * cosθi + n2 * cosθt)
            r_par = -1.0 * (n1 * cosθt - n2 * cosθi) / (n1 * cosθt + n2 * cosθi)
            F = (abs(r_per) ** 2 + abs(r_par) ** 2) / 2.0
            T = 1.0 - F

            # compute reflection rays
            reflected_ray_dir = (ray.dir - N * 2.0 * ray.dir.dot(N)).normalize()
            reflected_rays = Ray(
                nudged,
                reflected_ray_dir,
                ray.depth + 1,
                ray.n,
                ray.reflections + 1,
                ray.transmissions,
                ray.diffuse_reflections,
            )

            # compute refraction rays
            # Spectrum dispersion is not implemented.
            # We approximate refraction direction averaging index of refraction of each wavelenght
            n1_div_n2_aver = n1_div_n2.average()
            sin2θt = (n1_div_n2_aver) ** 2 * (1.0 - cosθi ** 2)

            non_TiR = sin2θt <= 1.0
            refracted_ray_dir = (
                ray.dir * (n1_div_n2_aver)
                + N * (n1_div_n2_aver * cosθi - np.sqrt(1 - np.clip(sin2θt, 0, 1)))
            ).normalize()
            nudged = hit.point - N * 0.000001  # nudged for refraction
            refracted_rays = Ray(
                nudged,
                refracted_ray_dir,
                ray.depth + 1,
                n2,
                ray.reflections,
                ray.transmissions + 1,
                ray.diffuse_reflections,
            )

            # compute color, either deterministically or using Monte Carlo
            if hit.surface.mc:
                # we lose the color-dependency of F because of MC
                pick = lambda randomness: Ray.where(
                    (randomness > F.average()) & non_TiR, refracted_rays, reflected_rays
                )
                rays = pick(np.random.rand(len(reflected_rays)))
                color = get_raycolor(rays, scene)
            else:
                color = get_raycolor(reflected_rays, scene) * F
                if np.any(non_TiR):
                    color += (
                        get_raycolor(refracted_rays.extract(non_TiR), scene).place(
                            non_TiR
                        )
                        * T
                    )

            # absorption:
            # approximation using wavelength for red = 630 nm, green 550 nm, blue 475 nm
            color = color * vec3.exp(
                -2.0
                * vec3.imag(ray.n)
                * 2.0
                * np.pi
                / vec3(630, 550, 475)
                * 1e9
                * hit.distance
            )
        return color
