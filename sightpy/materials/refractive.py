from ..utils.constants import *
from ..utils.vector3 import vec3, rgb, extract
from ..utils.random import normal_pdf
from functools import reduce as reduce
from ..ray import Ray, get_raycolor
from .. import lights
import numpy as np
from . import Material


class Refractive(Material):
    def __init__(
        self,
        n,
        n_ref=vec3(1.5 + 0.0j, 1.5 + 0.0j, 1.5 + 0.0j),
        purity=0.9,
        purity_ref=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n = n  # index of refraction
        self.n_ref = n_ref

        self.purity = purity  # purity of material. 1 is completely pure, 0 is complete randomness.
        self.purity_ref = (
            purity_ref  # The greater the purity, the narrower the distribution
        )
        #                               used to calculate the direction of the refracted ray.

        # Instead of defining a index of refraction (n) for each wavelenght (computationally expensive)
        # we aproximate defining the index of refraction
        # using a vec3 for red = 630 nm, green 555 nm, blue 475 nm, the most sensitive wavelenghts of human eye.

        # Index a refraction is a complex number.
        # The real part is involved in how much light is reflected and model refraction direction via Snell Law.
        # The imaginary part of n is involved in how much light is reflected and absorbed. For non-transparent materials like metals is usually between (0.1j,3j)
        # and for transparent materials like glass is  usually between (0.j , 1e-7j)

    def get_color(self, scene, ray, hit, max_index):
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

            def get_fresnel(n2):
                # compute complete fresnel term
                cosθt = vec3.sqrt(1.0 - (n1 / n2) ** 2 * (1.0 - cosθi ** 2))
                r_per = (n1 * cosθi - n2 * cosθt) / (n1 * cosθi + n2 * cosθt)
                r_par = -1.0 * (n1 * cosθt - n2 * cosθi) / (n1 * cosθt + n2 * cosθi)
                F = (r_per.abs() ** 2 + r_par.abs() ** 2) / 2.0
                return F

            cosθi = V.dot(N)
            n1 = ray.n
            n2 = vec3.where(hit.orientation == UPWARDS, self.n, scene.n)
            n2_ref = vec3.where(hit.orientation == UPWARDS, self.n_ref, scene.n)
            F, F_ref = get_fresnel(n2), get_fresnel(n2_ref)

            # compute reflection
            reflected_ray_dir = (ray.dir - N * 2.0 * ray.dir.dot(N)).normalize()
            reflected_ray_deps = ray.ray_index.reshape((ray.length, 1))

            # Compute ray directions and probabilities
            pdf = normal_pdf(reflected_ray_dir, 1 - self.purity)
            pdf_ref = normal_pdf(reflected_ray_dir, 1 - self.purity_ref)

            sampled_ray_dir = pdf.generate()
            log_PDF_val = pdf.log_value(sampled_ray_dir)
            log_PDF_val_ref = pdf_ref.log_value(sampled_ray_dir)

            np.seterr(divide="ignore")  # ignore log(0) computations
            new_log_p_z = ray.log_p_z + log_PDF_val + np.log(F.x)
            new_log_p_z_ref = ray.log_p_z_ref + log_PDF_val_ref + np.log(F_ref.x)
            np.seterr(divide="warn")  # unset warning ignore

            assert np.all((ray.log_p_z + np.log(F.x) < 1e-7) | (ray.log_p_z == 1.0))
            assert np.all(
                (ray.log_p_z_ref + np.log(F_ref.x) < 1e-7) | (ray.log_p_z == 1.0)
            )

            ray_reflect, max_index = get_raycolor(
                Ray(
                    pixel_index=ray.pixel_index,
                    ray_index=ray.ray_index,
                    ray_dependencies=np.hstack(
                        (ray.ray_dependencies, reflected_ray_deps)
                    ),
                    origin=nudged,
                    dir=sampled_ray_dir,
                    depth=ray.depth + 1,
                    n=ray.n,
                    log_trans_probs=new_log_p_z,
                    log_trans_probs_ref=new_log_p_z_ref,
                    color=ray.color,
                    reflections=ray.reflections + 1,
                    transmissions=ray.transmissions,
                    diffuse_reflections=ray.diffuse_reflections,
                ),
                scene,
                max_index,
            )

            # Update F to account for any extra rays picked up
            reflect_deps = ray_reflect.ray_dependencies[:, -1]
            ray_reflect.ray_dependencies = np.delete(
                ray_reflect.ray_dependencies, -1, axis=1
            )

            # want ray.index(pos) for pos in reflect_deps
            reflect_indexing_order = [
                # indices in original ray matching pos (there should be exactly 1):
                np.where(ray.ray_index == pos)[0][0]
                for pos in reflect_deps
            ]

            F_reflect = F.expand_by_index(reflect_indexing_order)
            F_ref_reflect = F_ref.expand_by_index(reflect_indexing_order)

            color_reflect = (
                color.repeat(ray_reflect.color.shape()[0])
                + ray_reflect.color * F_reflect
            )
            # color_reflect = color.repeat(ray_reflect.color.shape()[0]) + ray_reflect.color
            color_ref_reflect = (
                color.repeat(ray_reflect.color.shape()[0])
                + ray_reflect.color * F_ref_reflect
            )
            # color_ref_reflect = color.repeat(ray_reflect.color.shape()[0]) + ray_reflect.color

            # compute refraction rays
            # Spectrum dispersion is not implemented.
            # We approximate refraction direction averaging index of refraction of each wavelength
            def get_non_tir(n2):
                n1_div_n2 = vec3.real(n1) / vec3.real(n2)
                n1_div_n2_aver = n1_div_n2.average()
                sin2θt = (n1_div_n2_aver) ** 2 * (1.0 - cosθi ** 2)

                new_ray_dir = (
                    ray.dir * (n1_div_n2_aver)
                    + N * (n1_div_n2_aver * cosθi - np.sqrt(1 - np.clip(sin2θt, 0, 1)))
                ).normalize()
                return sin2θt <= 1.0, new_ray_dir

            non_TiR, refracted_ray_dir = get_non_tir(n2)
            non_TiR_ref, refracted_ray_dir_ref = get_non_tir(n2_ref)
            if np.any(non_TiR) or np.any(
                non_TiR_ref
            ):  # avoid total internal reflection
                nudged = hit.point - N * 0.000001  # nudged for refraction
                T = 1.0 - F
                T_ref = 1.0 - F_ref
                # fix rounding issues
                T.x[np.abs(T.x) < 1e-13] = 0.0
                T_ref.x[np.abs(T_ref.x) < 1e-13] = 0.0

                # Compute ray directions and probabilities
                pdf = normal_pdf(refracted_ray_dir, 1 - self.purity)
                pdf_ref = normal_pdf(refracted_ray_dir_ref, 1 - self.purity_ref)

                sampled_ray_dir = pdf.generate()
                log_PDF_val = pdf.log_value(sampled_ray_dir)
                log_PDF_val_ref = pdf_ref.log_value(sampled_ray_dir)

                np.seterr(divide="ignore")  # ignore log(0) computations
                new_log_p_z = ray.log_p_z + log_PDF_val + np.log(T.x)
                new_log_p_z_ref = ray.log_p_z_ref + log_PDF_val_ref + np.log(T_ref.x)
                np.seterr(divide="warn")  # unset warning ignore

                refracted_ray_indices = np.array(range(ray.length) + max_index + 1)
                refracted_ray_deps = refracted_ray_indices.reshape((ray.length, 1))

                assert np.all(ray.log_p_z != 1.0)
                assert np.all(ray.log_p_z_ref != 1.0)
                assert np.all((new_log_p_z < 1e-7) | (ray.log_p_z == 1.0))
                assert np.all((new_log_p_z_ref < 1e-7) | (ray.log_p_z_ref == 1.0))

                ray_refract, new_max_index = get_raycolor(
                    Ray(
                        ray.pixel_index,
                        refracted_ray_indices,
                        np.hstack((ray.ray_dependencies, refracted_ray_deps)),
                        nudged,
                        sampled_ray_dir,
                        ray.depth + 1,
                        n2,
                        new_log_p_z,
                        new_log_p_z_ref,
                        ray.color,
                        ray.reflections,
                        ray.transmissions + 1,
                        ray.diffuse_reflections,
                    ).extract(non_TiR),
                    scene,
                    ray.length + max_index + 1,
                )

                assert np.all(
                    (np.abs(ray_refract.log_p_z - ray_refract.log_p_z_ref) > 1e-7)
                    | (ray_refract.log_p_z == 1.0)
                    | (ray_refract.log_p_z == -np.inf)
                )
                assert not any(ray_reflect.log_p_z == -np.inf)
                assert not any(ray_reflect.log_p_z_ref == -np.inf)

                # update nonTiR with new copy order
                refract_indexing_order = [
                    index
                    for pos in ray_refract.ray_dependencies[:, -1]
                    for index in np.where(range(ray.length) + max_index + 1 == pos)[0]
                ]
                ray_refract.ray_dependencies = np.delete(
                    ray_refract.ray_dependencies, -1, axis=1
                )

                non_TiR = np.array(
                    [non_TiR[round(pos)] for pos in refract_indexing_order]
                )
                non_TiR_ref = np.array(
                    [non_TiR_ref[round(pos)] for pos in refract_indexing_order]
                )
                T = T.expand_by_index(refract_indexing_order)
                T_ref = T_ref.expand_by_index(refract_indexing_order)

                refracted_color = ray_refract.color * T.extract(non_TiR)
                refracted_color_ref = ray_refract.color * T_ref.extract(non_TiR_ref)

                color_refract = color.repeat(
                    ray_refract.color.shape()[0]
                ) + refracted_color.place(non_TiR)
                color_ref_refract = color.repeat(
                    ray_refract.color.shape()[0]
                ) + refracted_color_ref.place(non_TiR_ref)

                # Record reflected and refracted ray colors separately
                color = color_reflect.append(color_refract)
                color_ref = color_ref_reflect.append(color_ref_refract)

                ray_out = ray_reflect.combine(ray_refract)

            else:  # not TIR
                color = color_reflect
                color_ref = color_ref_reflect
                ray_out = ray_reflect

            # absorption:
            # approximation using wavelength for red = 630 nm, green 550 nm, blue 475 nm
            full_indexing_order = reflect_indexing_order + refract_indexing_order
            hit_distance_repeated = np.array(
                [hit.distance[round(pos)] for pos in full_indexing_order]
            )
            ray_n_repeated = n1.expand_by_index(full_indexing_order)

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
            color_match = (ray_out.color - color_ref).abs() < 1e-6

            assert all(color_match)

            assert np.all(
                (np.abs(ray_out.log_p_z - ray_out.log_p_z_ref) > 1e-7)
                | (ray_out.log_p_z == 1.0)
                | (ray_out.log_p_z == -np.inf)
            )

            return ray_out

        else:  # Too deep and didn't hit a light source, return impossible logprob = 1
            n_rays = ray.log_p_z.shape[0]
            ray.log_p_z = np.full(n_rays, 1)
            ray.log_p_z_ref = np.full(n_rays, 1)
            ray.color = color.repeat(n_rays)
            return ray
