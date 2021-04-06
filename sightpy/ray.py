from .utils.constants import *
from .utils.vector3 import vec3, extract, rgb
import numpy as np
from functools import reduce as reduce


class Ray:
    """Info of the ray and the media it's travelling.
    Note that we can encode a series of individual rays within this class."""

    def __init__(
        self,
        pixel_index,
        ray_index,
        ray_dependencies,
        origin,
        dir,
        depth,
        n,
        log_trans_probs,
        log_trans_probs_ref,
        color,
        reflections,
        transmissions,
        diffuse_reflections,
    ):
        self.length = max(len(origin), len(dir), len(n))
        shape = [self.length]

        self.pixel_index = pixel_index  # keep track of which pixel this ray belongs to.
        self.ray_index = ray_index  # Each ray has a unique index
        self.ray_dependencies = (
            ray_dependencies  # keep track of inter-ray dependencies.
        )

        self.origin = origin  # the point where the ray comes from
        self.dir = dir  # direction of the ray
        self.depth = depth  # ray_depth is the number of the reflections + transmissions/refractions,
        #                     # starting at zero for camera rays
        self.n = n.broadcast_to(
            shape
        )  # ray_n is the index of refraction of the media in which the ray is travelling

        # Record the probability of each ray's trajectory.
        # This will be zero if it never hits a light source.
        self.log_p_z = log_trans_probs
        self.log_p_z_ref = log_trans_probs_ref
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
        self.diffuse_reflections = (
            diffuse_reflections  # reflections is the number of the refrections,
        )
        #                                               # starting at zero for camera rays

    def extract(self, hit_check):
        # ray_dependencies is a 2d array, so adjust hit_check accordingly
        target_shape = sum(hit_check), self.ray_dependencies.shape[1]
        hit_check_dep = np.reshape(hit_check, (self.length, 1))
        hit_check_dep = hit_check_dep.repeat(target_shape[1], axis=1)
        return Ray(
            np.extract(hit_check, self.pixel_index),
            np.extract(hit_check, self.ray_index),
            np.extract(hit_check_dep, self.ray_dependencies).reshape(target_shape),
            self.origin.extract(hit_check),
            self.dir.extract(hit_check),
            self.depth,
            self.n.extract(hit_check),
            np.extract(hit_check, self.log_p_z),
            np.extract(hit_check, self.log_p_z_ref),
            self.color.extract(hit_check),
            self.reflections,
            self.transmissions,
            self.diffuse_reflections,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        return Ray(
            self.pixel_index[ind],
            self.ray_index[ind],
            self.ray_dependencies[ind],
            self.origin.get_subvec(ind),
            self.dir.get_subvec(ind),
            self.depth,
            self.n.get_subvec(ind),
            self.log_p_z[ind],
            self.log_p_z_ref[ind],
            self.color.get_subvec(ind),
            self.reflections,
            self.transmissions,
            self.diffuse_reflections,
        )

    @staticmethod
    def where(cond, x, y):
        if x.depth != y.depth:
            raise ValueError("Both rays must have same depth")
        return Ray(
            np.where(cond, x.pixel_index, y.pixel_index),
            np.where(cond, x.ray_index, y.ray_index),
            np.where(cond, x.ray_dependencies, y.ray_dependencies),
            vec3.where(cond, x.origin, y.origin),
            vec3.where(cond, x.dir, y.dir),
            x.depth,
            vec3.where(cond, x.n, y.n),
            np.where(cond, x.log_p_z, y.log_p_z),
            np.where(cond, x.log_p_z_ref, y.log_p_z_ref),
            vec3.where(cond, x.color, y.color),
            max(x.reflections, y.reflections),
            max(x.transmissions, y.transmissions),
            max(x.diffuse_reflections, y.diffuse_reflections),
        )

    @staticmethod
    def concatenate(rays):
        pixel_index = [r.pixel_index for r in rays]
        ray_index = [r.ray_index for r in rays]
        ray_dependencies = [r.ray_dependencies for r in rays]
        origin = [r.origin for r in rays]
        dir = [r.dir for r in rays]
        depth = rays[0].depth
        n = [r.n for r in rays]
        log_p_z = [r.log_p_z for r in rays]
        log_p_z_ref = [r.log_p_z_ref for r in rays]
        color = [r.color for r in rays]
        reflections = max(r.reflections for r in rays)
        transmissions = max(r.transmissions for r in rays)
        diffuse_reflections = max(r.diffuse_reflections for r in rays)

        if not all(r.depth == depth for r in rays):
            print("All rays must have same depth!")
        return Ray(
            np.concatenate(pixel_index),
            np.concatenate(ray_index),
            np.vstack(ray_dependencies),
            vec3.concatenate(origin),
            vec3.concatenate(dir),
            depth,
            vec3.concatenate(n),
            np.concatenate(log_p_z),
            np.concatenate(log_p_z_ref),
            vec3.concatenate(color),
            reflections,
            transmissions,
            diffuse_reflections,
        )

    def place(self, mask, other):
        # ray_dependencies is a 2d array, so adjust hit_check accordingly
        target_shape = sum(mask), self.ray_dependencies.shape[1]
        mask_dep = np.reshape(mask, (self.length, 1))
        mask_dep = mask_dep.repeat(target_shape[1], axis=1)

        self.origin = self.origin.place_into(mask, other.origin)
        self.dir = self.dir.place_into(mask, other.dir)
        # self.n = self.n.place_into(mask, other.n)
        self.color = self.color.place_into(mask, other.color)
        np.place(self.pixel_index, mask, other.pixel_index)
        np.place(self.ray_index, mask, other.ray_index)
        np.place(self.ray_dependencies, mask_dep, other.ray_dependencies.reshape(-1))
        np.place(self.log_p_z, mask, other.log_p_z)
        np.place(self.log_p_z_ref, mask, other.log_p_z_ref)

    def combine(self, other: "Ray"):
        """Merge two sets of rays into one."""
        assert self.ray_dependencies.shape[1] == other.ray_dependencies.shape[1]
        return Ray(
            np.concatenate((self.pixel_index, other.pixel_index)),
            np.concatenate((self.ray_index, other.ray_index)),
            np.vstack((self.ray_dependencies, other.ray_dependencies)),
            self.origin.append(other.origin),
            self.dir.append(other.dir),
            max(self.depth, other.depth),
            self.n.append(other.n),
            np.concatenate((self.log_p_z, other.log_p_z)),
            np.concatenate((self.log_p_z_ref, other.log_p_z_ref)),
            self.color.append(other.color),
            max(self.reflections, other.reflections),
            max(self.transmissions, other.transmissions),
            max(self.diffuse_reflections, other.diffuse_reflections),
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


def get_raycolor(ray, scene, max_index=0):
    # Compute all collisions for each ray
    inters = [s.intersect(ray.origin, ray.dir) for s in scene.collider_list]
    distances, hit_orientation = zip(*inters)

    # get the shortest distance collision
    nearest = reduce(np.minimum, distances)

    ray_out = Ray(
        ray.pixel_index,
        ray.ray_index,
        ray.ray_dependencies,
        ray.origin,
        ray.dir,
        ray.depth,
        ray.n,
        ray.log_p_z,
        ray.log_p_z_ref,
        ray.color,
        ray.reflections,
        ray.transmissions,
        ray.diffuse_reflections,
    )
    # Initialise if necessary
    if max_index == 0:
        max_index = max(ray_out.ray_index)

    for (coll, dis, orient) in zip(scene.collider_list, distances, hit_orientation):
        hit_check = (nearest != FARAWAY) & (dis == nearest)

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

            sub_rays = material.get_color(
                scene, ray.extract(hit_check), hit_info, max_index
            )

            # Recombine the rays into the current one.
            # First place the original rays into the main ray
            original_rays = np.extract(hit_check, ray.ray_index)
            original_mask = np.isin(sub_rays.ray_index, original_rays)
            n_extra_rays = ray_out.length - ray.length
            if n_extra_rays > 0:
                full_hit_check = np.concatenate(
                    (hit_check, np.full(n_extra_rays, False))
                )
            else:
                full_hit_check = hit_check

            ray_out.place(
                full_hit_check, sub_rays.extract(original_mask),
            )
            # Then add the rest to the end
            ray_out = ray_out.combine(sub_rays.extract(np.logical_not(original_mask)))

            # update max index if necessary
            max_index = max(max_index, max(sub_rays.ray_index))

    return ray_out, max_index


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
