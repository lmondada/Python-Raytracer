from .utils.constants import *
from .utils.vector3 import vec3, extract, rgb
import numpy as np
from functools import reduce as reduce


class Ray:
    """Info of the ray and the media it's travelling"""

    def __init__(
        self, origin, dir, depth, n, reflections, transmissions, diffuse_reflections
    ):
        self.length = max(len(origin), len(dir), len(n))
        shape = [self.length]

        self.origin = origin.broadcast_to(shape)  # the point where the ray comes from
        self.dir = dir.broadcast_to(shape)  # direction of the ray
        self.depth = depth  # ray_depth is the number of the refrections + transmissions/refractions, starting at zero for camera rays
        self.n = n.broadcast_to(
            shape
        )  # ray_n is the index of refraction of the media in which the ray is travelling

        # Instead of defining a index of refraction (n) for each wavelenght (computationally expensive) we aproximate defining the index of refraction
        # using a vec3 for red = 630 nm, green 555 nm, blue 475 nm, the most sensitive wavelenghts of human eye.

        # Index a refraction is a complex number.
        # The real part is involved in how much light is reflected and model refraction direction via Snell Law.
        # The imaginary part of n is involved in how much light is reflected and absorbed. For non-transparent materials like metals is usually between (0.1j,3j)
        # and for transparent materials like glass is  usually between (0.j , 1e-7j)

        self.reflections = reflections  # reflections is the number of the refrections, starting at zero for camera rays
        self.transmissions = transmissions  # transmissions is the number of the transmissions/refractions, starting at zero for camera rays
        self.diffuse_reflections = diffuse_reflections  # reflections is the number of the refrections, starting at zero for camera rays

    def extract(self, hit_check):
        return Ray(
            self.origin.extract(hit_check),
            self.dir.extract(hit_check),
            self.depth,
            self.n.extract(hit_check),
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
            reflections,
            transmissions,
            diffuse_reflections,
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

    inters = [s.intersect(ray.origin, ray.dir) for s in scene.collider_list]
    distances, hit_orientation = zip(*inters)

    # get the shortest distance collision
    nearest = reduce(np.minimum, distances)
    color = rgb(0.0, 0.0, 0.0)

    for (coll, dis, orient) in zip(scene.collider_list, distances, hit_orientation):
        hit_check = (nearest != FARAWAY) & (dis == nearest)

        if np.any(hit_check):

            material = coll.assigned_primitive.material
            hit_info = Hit(
                extract(hit_check, dis),
                extract(hit_check, orient),
                material,
                coll,
                coll.assigned_primitive,
            )

            cc = material.get_color(scene, ray.extract(hit_check), hit_info)
            color += cc.place(hit_check)

    return color


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
