from ..utils.constants import *
from ..utils.vector3 import vec3, rgb, extract
from functools import reduce as reduce
from ..ray import Ray, get_raycolor
from .. import lights
import numpy as np
from . import Material


class Refractive(Material):
    def __init__(self, n, n_ref=vec3(1.5 + 0.j, 1.5 + 0.j, 1.5 + 0.j), **kwargs):
        super().__init__(**kwargs)

        self.n = n  # index of refraction
        self.n_ref = n_ref

        # Instead of defining a index of refraction (n) for each wavelenght (computationally expensive)
        # we aproximate defining the index of refraction
        # using a vec3 for red = 630 nm, green 555 nm, blue 475 nm, the most sensitive wavelenghts of human eye.

        # Index a refraction is a complex number.
        # The real part is involved in how much light is reflected and model refraction direction via Snell Law.
        # The imaginary part of n is involved in how much light is reflected and absorbed. For non-transparent materials like metals is usually between (0.1j,3j)
        # and for transparent materials like glass is  usually between (0.j , 1e-7j)

    def get_color(self, scene, ray, hit):
        hit.point = ray.origin + ray.dir * hit.distance  # intersection point
        N = hit.material.get_Normal(hit)  # normal

        color = rgb(0.0, 0.0, 0.0)
        color_ref = rgb(0.0, 0.0, 0.0)

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
            def get_fresnel(n2):
                # compute complete fresnel term
                cosθt = vec3.sqrt(1. - (n1 / n2) ** 2 * (1. - cosθi ** 2))
                r_per = (n1 * cosθi - n2 * cosθt) / (n1 * cosθi + n2 * cosθt)
                r_par = -1. * (n1 * cosθt - n2 * cosθi) / (n1 * cosθt + n2 * cosθi)
                F = (r_per.abs() ** 2 + r_par.abs() ** 2) / 2.
                return F

            n_rays = ray.p_z.shape[0]
            cosθi = V.dot(N)
            n1 = ray.n
            n2 = vec3.where(hit.orientation == UPWARDS, self.n, scene.n)
            n2_ref = vec3.where(hit.orientation == UPWARDS, self.n_ref, scene.n)
            F, F_ref = get_fresnel(n2), get_fresnel(n2_ref)

            # compute reflection
            reflected_ray_dir = (ray.dir - N * 2. * ray.dir.dot(N)).normalize()
            ray_reflect, copy_order_reflect = get_raycolor(
                Ray(
                    nudged,
                    reflected_ray_dir,
                    ray.depth + 1,
                    ray.n,
                    ray.p_z,
                    ray.p_z_ref,
                    ray.color,
                    ray.reflections + 1,
                    ray.transmissions,
                    ray.diffuse_reflections
                ), scene
            )
            if len(copy_order_reflect) > 0:
                F_reflect = F.append(tuple([
                    F.splice(round(pos), round(pos)+1) for pos in copy_order_reflect
                ]))
                F_ref_reflect = F_ref.append(tuple([
                    F_ref.splice(round(pos), round(pos)+1) for pos in copy_order_reflect
                ]))
            else:
                F_reflect, F_ref_reflect = F, F_ref
            color_reflect = color.repeat(ray_reflect.color.shape()[0]) + ray_reflect.color * F_reflect
            color_ref_reflect = color.repeat(ray_reflect.color.shape()[0]) + ray_reflect.color * F_ref_reflect

            # compute refraction rays
            # Spectrum dispersion is not implemented.
            # We approximate refraction direction averaging index of refraction of each wavelength
            def get_non_tir(n2):
                n1_div_n2 = vec3.real(n1) / vec3.real(n2)
                n1_div_n2_aver = n1_div_n2.average()
                sin2θt = (n1_div_n2_aver) ** 2 * (1. - cosθi ** 2)

                new_ray_dir = (
                        ray.dir * (n1_div_n2_aver) + N * (
                        n1_div_n2_aver * cosθi - np.sqrt(1 - np.clip(sin2θt, 0, 1))
                )
                ).normalize()
                return sin2θt <= 1., new_ray_dir

            non_TiR, refracted_ray_dir = get_non_tir(n2)
            non_TiR_ref, refracted_ray_dir_ref = get_non_tir(n2_ref)
            n_refracted = 0
            if np.any(non_TiR) or np.any(non_TiR_ref):  # avoid total internal reflection
                nudged = hit.point - N * .000001  # nudged for refraction
                T = 1. - F
                T_ref = 1. - F_ref

                ray_refract, copy_order_refract = get_raycolor(
                    Ray(
                        nudged,
                        refracted_ray_dir,
                        ray.depth + 1,
                        n2,
                        ray.p_z,
                        ray.p_z_ref,
                        ray.color,
                        ray.reflections,
                        ray.transmissions + 1,
                        ray.diffuse_reflections,
                    ).extract(non_TiR),
                    scene
                )
                n_refracted = ray_refract.p_z.shape[0]
                # Changing n will change where this ray goes => must have probability zero for theta_ref.
                if self.n != self.n_ref:
                    ray_refract.p_z_ref = np.clip(ray_refract.p_z_ref, None, 0)

                # update nonTiR with new copy order
                if len(copy_order_refract) > 0:
                    non_TiR = np.concatenate((
                        non_TiR,
                        [non_TiR[round(pos)] for pos in copy_order_refract]
                    ))
                    non_TiR_ref = np.concatenate((
                        non_TiR_ref,
                        [non_TiR_ref[round(pos)] for pos in copy_order_refract]
                    ))
                    T = T.append(tuple([
                        T.splice(round(pos), round(pos) + 1) for pos in copy_order_refract
                    ]))
                    T_ref = T_ref.append(tuple([
                        T_ref.splice(round(pos), round(pos) + 1) for pos in copy_order_refract
                    ]))
                refracted_color = ray_refract.color * T.extract(non_TiR)
                refracted_color_ref = ray_refract.color * T_ref.extract(non_TiR_ref)

                color_refract = color.repeat(ray_refract.color.shape()[0]) + refracted_color.place(non_TiR)
                color_ref_refract = color.repeat(ray_refract.color.shape()[0]) + refracted_color_ref.place(non_TiR_ref)

                # Record reflected and refracted ray colors separately
                color = color_reflect.append(color_refract)
                color_ref = color_ref_reflect.append(color_ref_refract)

                ray_out = ray_reflect.combine(ray_refract)

            else:  # not TIR
                copy_order_refract = []
                color = color_reflect
                color_ref = color_ref_reflect
                ray_out = ray_reflect

            # absorption:
            # approximation using wavelength for red = 630 nm, green 550 nm, blue 475 nm
            hit_distance_repeated = np.concatenate((
                hit.distance,
                [hit.distance[round(pos)] for pos in copy_order_reflect],
                hit.distance if n_refracted > 0 else [],
                [hit.distance[round(pos)] for pos in copy_order_refract]
            ))
            if n1.shape()[0] == 1:
                ray_n_repeated = n1
            else:
                ray_n_repeated = n1.append((
                    *(n1.splice(round(pos), round(pos)+1) for pos in copy_order_reflect),
                    n1 if n_refracted > 0 else [],
                    *(n1.splice(round(pos), round(pos)+1) for pos in copy_order_refract)
                ))
            ambient_factor = vec3.exp(
                -2.0
                * vec3.imag(ray_n_repeated)
                * 2.0
                * np.pi
                / vec3(630, 550, 475)
                * 1e9
                * hit_distance_repeated
            )
            ray_out.color = color * ambient_factor
            color_ref = color_ref * ambient_factor

            # Update ray color probabilities
            color_match = ray_out.color == color_ref
            ray_out.p_z_ref = ray_out.p_z_ref * color_match

            full_copy_order = [
                *copy_order_reflect,
                *(list(range(n_rays)) if n_refracted > 0 else []),
                *copy_order_refract
            ]
            return ray_out, full_copy_order

        else:  # Too deep and didn't hit a light source, return negative probabilities.
            n_rays = ray.p_z.shape[0]
            ray.p_z = np.full(n_rays, -1)
            ray.p_z_ref = np.full(n_rays, -1)
            ray.color = color.repeat(n_rays)
            return ray, []
