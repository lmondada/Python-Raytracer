from .utils.constants import *
from .utils.vector3 import vec3, extract, rgb
import numpy as np
from functools import reduce as reduce


class Ray:
    """Info of the ray and the media it's travelling.
    Note that we can encode a series of individual rays within this class."""

    def __init__(
            self, origin, dir, depth, n, trans_probs, trans_probs_ref, color,
            reflections, transmissions, diffuse_reflections
        ):
        self.length = max(len(origin), len(dir), len(n))
        shape = [self.length]

        self.origin = origin  # the point where the ray comes from
        self.dir = dir        # direction of the ray
        self.depth = depth    # ray_depth is the number of the reflections + transmissions/refractions,
        #                     # starting at zero for camera rays
        self.n = n.broadcast_to(
            shape
        )                     # ray_n is the index of refraction of the media in which the ray is travelling

        # Record the probability of each ray's trajectory.
        # This will be zero if it never hits a light source.
        self.p_z = trans_probs
        self.p_z_ref = trans_probs_ref
        # keep track of the color of each sub ray.
        self.color = color

        # Instead of defining a index of refraction (n) for each wavelenght (computationally expensive) we aproximate
        # defining the index of refraction
        # using a vec3 for red = 630 nm, green 555 nm, blue 475 nm, the most sensitive wavelenghts of human eye.

        # Index a refraction is a complex number.
        # The real part is involved in how much light is reflected and model refraction direction via Snell Law.
        # The imaginary part of n is involved in how much light is reflected and absorbed. For non-transparent
        # materials like metals is usually between (0.1j,3j)
        # and for transparent materials like glass is  usually between (0.j , 1e-7j)

        self.reflections = reflections  # reflections is the number of the refrections, starting at zero for camera rays
        self.transmissions = transmissions  # transmissions is the number of the transmissions/refractions,
        #                                   # starting at zero for camera rays
        self.diffuse_reflections = diffuse_reflections  # reflections is the number of the refrections,
        #                                               # starting at zero for camera rays

    def extract(self, hit_check):
        return Ray(
            self.origin.extract(hit_check),
            self.dir.extract(hit_check),
            self.depth,
            self.n.extract(hit_check),
            np.extract(hit_check, self.p_z),
            np.extract(hit_check, self.p_z_ref),
            self.color.extract(hit_check),
            self.reflections,
            self.transmissions,
            self.diffuse_reflections,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        return Ray(
            self.origin[ind],
            self.dir[ind],
            self.depth,
            self.n[ind],
            self.p_z[ind],
            self.p_z_ref[ind],
            self.color[ind],
            self.reflections,
            self.transmissions,
            self.diffuse_reflections,
        )

    @staticmethod
    def where(cond, x, y):
        if x.depth != y.depth:
            raise ValueError("Both rays must have same depth")
        return Ray(
            vec3.where(cond, x.origin, y.origin),
            vec3.where(cond, x.dir, y.dir),
            x.depth,
            vec3.where(cond, x.n, y.n),
            np.where(cond, x.p_z, y.p_z),
            np.where(cond, x.p_z_ref, y.p_z_ref),
            vec3.where(cond, x.color, y.color),
            max(x.reflections, y.reflections),
            max(x.transmissions, y.transmissions),
            max(x.diffuse_reflections, y.diffuse_reflections),
        )

    @staticmethod
    def concatenate(rays):
        origin = [r.origin for r in rays]
        dir = [r.dir for r in rays]
        depth = rays[0].depth
        n = [r.n for r in rays]
        p_z = [r.p_z for r in rays]
        p_z_ref = [r.p_z_ref for r in rays]
        color = [r.color for r in rays]
        reflections = max(r.reflections for r in rays)
        transmissions = max(r.transmissions for r in rays)
        diffuse_reflections = max(r.diffuse_reflections for r in rays)

        if not all(r.depth == depth for r in rays):
            print("All rays must have same depth!")
        return Ray(
            vec3.concatenate(origin),
            vec3.concatenate(dir),
            depth,
            vec3.concatenate(n),
            np.concatenate(p_z),
            np.concatenate(p_z_ref),
            vec3.concatenate(color),
            reflections,
            transmissions,
            diffuse_reflections,
        )

    def combine(self, other: "Ray"):
        """Merge two sets of rays into one."""
        return Ray(
            self.origin.append(other.origin),
            self.dir.append(other.dir),
            max(self.depth, other.depth),
            self.n.append(other.n),
            np.concatenate((self.p_z, other.p_z)),
            np.concatenate((self.p_z_ref, other.p_z_ref)),
            self.color.append(other.color),
            max(self.reflections, other.reflections),
            max(self.transmissions, other.transmissions),
            max(self.diffuse_reflections, other.diffuse_reflections)
        )


class Hit:
    """Info of the ray-surface intersection"""

    def __init__(self, distance, orientation, material, collider, surface):
        self.distance = distance
        self.orientation = orientation
        self.material = material
        self.collider = collider
        self.surface = surface
        self.u = None
        self.v = None
        self.N = None
        self.point = None

    def get_uv(self):
        if self.u is None:  # this is for prevent multiple computations of u,v
            self.u, self.v = self.collider.assigned_primitive.get_uv(self)
        return self.u, self.v

    def get_normal(self):
        if self.N is None:  # this is for prevent multiple computations of normal
            self.N = self.collider.get_N(self)
        return self.N


def get_raycolor(ray, scene):
    # Compute all collisions for each ray
    inters = [s.intersect(ray.origin, ray.dir) for s in scene.collider_list]
    distances, hit_orientation = zip(*inters)

    # get the shortest distance collision
    nearest = reduce(np.minimum, distances)

    # Since we are keeping track of the rays on the way down too, need to update hit check array.
    expanded_hit_check = []
    ray_out = Ray(
        ray.origin, ray.dir, ray.depth, ray.n,
        ray.p_z, ray.p_z_ref, ray.color,
        ray.reflections, ray.transmissions, ray.diffuse_reflections
    )

    def expand_hit_check(base, expansion):
        return np.concatenate((
            base,
            [base[round(pos)] for pos in expansion]
        ))

    for (coll, dis, orient) in zip(scene.collider_list, distances, hit_orientation):
        base_hit_check = (nearest != FARAWAY) & (dis == nearest)
        hit_check = expand_hit_check(base_hit_check, expanded_hit_check)

        # If this is the nearest for any ray, bounce that ray.
        if np.any(hit_check):
            material = coll.assigned_primitive.material
            hit_info = Hit(
                extract(hit_check, dis),
                extract(hit_check, orient),
                material,
                coll,
                coll.assigned_primitive,
            )

            sub_rays, copy_order = material.get_color(scene, ray.extract(hit_check), hit_info)

            # Recombine the rays into the current one.
            # We only really care about the color, probabilities and values for hit_check on our way back up.
            # Update the ray colors
            temp_col = sub_rays.color.place(hit_check)
            ray_out.color += temp_col

            # First rewrite the rays we investigated initially.
            np.place(ray_out.p_z, hit_check, sub_rays.p_z)
            np.place(ray_out.p_z_ref, hit_check, sub_rays.p_z_ref)

            # The duplicated rays share properties with their parents
            # Keep a list of references for each new ray.
            # Then, if we added extra rays combine these with the current ray.
            n_expected_rays = round(np.sum(hit_check))
            n_sub_rays = sub_rays.p_z.shape[0]
            n_added_rays = n_sub_rays - n_expected_rays

            if n_added_rays > 0:
                expanded_hit_check = np.concatenate((
                    expanded_hit_check,
                    # Integrate the previous copy order into the full ray order.
                    np.array([np.nonzero(hit_check)[0][round(pos)] for pos in copy_order])
                ))
                ray_out = ray_out.combine(sub_rays.extract([i >= n_expected_rays for i in range(n_sub_rays)]))

    return ray_out, expanded_hit_check


def get_distances(
    ray, scene
):  # Used for debugging ray-surface collisions. Return a grey map of objects distances.

    inters = [s.intersect(ray.origin, ray.dir) for s in scene.collider_list]
    distances, hit_orientation = zip(*inters)
    # get the shortest distance collision
    nearest = reduce(np.minimum, distances)

    max_r_distance = 10
    r_distance = np.where(nearest <= max_r_distance, nearest, max_r_distance)
    norm_r_distance = r_distance / max_r_distance
    return rgb(norm_r_distance, norm_r_distance, norm_r_distance)
