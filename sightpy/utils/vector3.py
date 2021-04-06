import numpy as np
import numbers
from typing import Tuple


def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        # Used for debugging. This method is called when you print an instance
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def __add__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x + v.x, self.y + v.y, self.z + v.z)
        elif isinstance(v, numbers.Number) or isinstance(v, np.ndarray):
            return vec3(self.x + v, self.y + v, self.z + v)

    def __radd__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x + v.x, self.y + v.y, self.z + v.z)
        elif isinstance(v, numbers.Number) or isinstance(v, np.ndarray):
            return vec3(self.x + v, self.y + v, self.z + v)

    def __sub__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x - v.x, self.y - v.y, self.z - v.z)
        elif isinstance(v, numbers.Number) or isinstance(v, np.ndarray):
            return vec3(self.x - v, self.y - v, self.z - v)

    def __rsub__(self, v):
        if isinstance(v, vec3):
            return vec3(v.x - self.x, v.y - self.y, v.z - self.z)
        elif isinstance(v, numbers.Number) or isinstance(v, np.ndarray):
            return vec3(v - self.x, v - self.y, v - self.z)

    def __mul__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x * v.x, self.y * v.y, self.z * v.z)
        elif isinstance(v, numbers.Number) or isinstance(v, np.ndarray):
            return vec3(self.x * v, self.y * v, self.z * v)

    def __rmul__(self, v):
        if isinstance(v, vec3):
            return vec3(v.x * self.x, v.y * self.y, v.z * self.z)
        elif isinstance(v, numbers.Number) or isinstance(v, np.ndarray):
            return vec3(v * self.x, v * self.y, v * self.z)

    def __truediv__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x / v.x, self.y / v.y, self.z / v.z)
        elif isinstance(v, numbers.Number) or isinstance(v, np.ndarray):
            return vec3(self.x / v, self.y / v, self.z / v)

    def __rtruediv__(self, v):
        if isinstance(v, vec3):
            return vec3(v.x / self.x, v.y / self.y, v.z / self.z)
        elif isinstance(v, numbers.Number) or isinstance(v, np.ndarray):
            return vec3(v / self.x, v / self.y, v / self.z)

    def abs(self):
        return vec3(np.abs(self.x), np.abs(self.y), np.abs(self.z))

    def __abs__(self):
        return self.abs()

    @staticmethod
    def real(v):
        return vec3(np.real(v.x), np.real(v.y), np.real(v.z))

    @staticmethod
    def imag(v):
        return vec3(np.imag(v.x), np.imag(v.y), np.imag(v.z))

    def yzx(self):
        return vec3(self.y, self.z, self.x)

    def xyz(self):
        return vec3(self.x, self.y, self.z)

    def zxy(self):
        return vec3(self.z, self.x, self.y)

    def average(self):
        return (self.x + self.y + self.z) / 3

    def matmul(self, matrix):
        if isinstance(self.x, numbers.Number):
            return array_to_vec3(np.dot(matrix, self.to_array()))
        elif isinstance(self.x, np.ndarray):
            return array_to_vec3(np.tensordot(matrix, self.to_array(), axes=([1, 0])))

    def change_basis(self, new_basis):
        return vec3(
            self.dot(new_basis[0]), self.dot(new_basis[1]), self.dot(new_basis[2])
        )

    def __pow__(self, a):
        return vec3(self.x ** a, self.y ** a, self.z ** a)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    @staticmethod
    def exp(v):
        return vec3(np.exp(v.x), np.exp(v.y), np.exp(v.z))

    @staticmethod
    def sqrt(v):
        return vec3(np.sqrt(v.x), np.sqrt(v.y), np.sqrt(v.z))

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    def cross(self, v):
        return vec3(
            self.y * v.z - self.z * v.y,
            -self.x * v.z + self.z * v.x,
            self.x * v.y - self.y * v.x,
        )

    def length(self):
        return np.sqrt(self.dot(self))

    def __len__(self):
        shape = self.shape()
        try:
            return shape[0]
        except TypeError:
            return shape

    # todo: do we need this?
    # def __getitem__(self, ind):
    #     return vec3(self.x[ind], self.y[ind], self.z[ind])

    def get_subvec(self, ind, ind_to=None):
        if ind_to is None:
            return vec3(self.x[ind], self.y[ind], self.z[ind])
        else:
            return vec3(self.x[ind:ind_to], self.y[ind:ind_to], self.z[ind:ind_to])

    def broadcast_to(self, shape):
        return vec3(
            np.broadcast_to(self.x, shape),
            np.broadcast_to(self.y, shape),
            np.broadcast_to(self.z, shape),
        )

    @staticmethod
    def concatenate(vecs):
        x = [vec.x for vec in vecs]
        y = [vec.y for vec in vecs]
        z = [vec.z for vec in vecs]
        return vec3(np.concatenate(x), np.concatenate(y), np.concatenate(z))

    def square_length(self):
        return self.dot(self)

    def normalize(self):
        mag = self.length()
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)

    def extract(self, cond):
        return vec3(extract(cond, self.x), extract(cond, self.y), extract(cond, self.z))

    @staticmethod
    def where(cond, out_true, out_false):
        return vec3(
            np.where(cond, out_true.x, out_false.x),
            np.where(cond, out_true.y, out_false.y),
            np.where(cond, out_true.z, out_false.z),
        )

    @staticmethod
    def select(mask_list, out_list):
        out_list_x = [i.x for i in out_list]
        out_list_y = [i.y for i in out_list]
        out_list_z = [i.z for i in out_list]

        return vec3(
            np.select(mask_list, out_list_x),
            np.select(mask_list, out_list_y),
            np.select(mask_list, out_list_z),
        )

    def clip(self, min, max):
        return vec3(
            np.clip(self.x, min, max),
            np.clip(self.y, min, max),
            np.clip(self.z, min, max),
        )

    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r

    def copy(self):
        return vec3(self.x, self.y, self.z,)

    def place_into(self, cond, vec):
        r = self.copy()
        np.place(r.x, cond, vec.x)
        np.place(r.y, cond, vec.y)
        np.place(r.z, cond, vec.z)
        return r

    def repeat(self, n):
        return vec3(np.repeat(self.x, n), np.repeat(self.y, n), np.repeat(self.z, n))

    def reshape(self, *newshape):
        return vec3(
            self.x.reshape(*newshape),
            self.y.reshape(*newshape),
            self.z.reshape(*newshape),
        )

    def shape(self, *newshape):
        if isinstance(self.x, numbers.Number):
            return 1
        elif isinstance(self.x, np.ndarray):
            return self.x.shape

    def append(self, vs):
        def ensure_is_list(v):
            if not isinstance(self.x, np.ndarray):
                v.x = [v.x]
                v.y = [v.y]
                v.z = [v.z]
            return v

        # ensure self has a list format.
        ensure_is_list(self)

        if isinstance(vs, vec3):
            return vec3(
                np.concatenate((self.x, vs.x)),
                np.concatenate((self.y, vs.y)),
                np.concatenate((self.z, vs.z)),
            )
        if isinstance(vs, Tuple) and len(vs) > 0:
            vs = tuple(map(ensure_is_list, vs))
            return vec3(
                np.concatenate((self.x, *(v.x for v in vs if isinstance(v, vec3)))),
                np.concatenate((self.y, *(v.y for v in vs if isinstance(v, vec3)))),
                np.concatenate((self.z, *(v.z for v in vs if isinstance(v, vec3)))),
            )

    def splice(self, i: int = None, j: int = None):
        return vec3(self.x[i:j], self.y[i:j], self.z[i:j],)

    def expand_by_index(self, indexing_order):
        vec = self.splice(
            round(indexing_order[0]), round(indexing_order[0]) + 1
        )  # empty vec3
        if len(indexing_order) > 1:
            vec = vec.append(
                tuple(
                    [
                        self.splice(round(pos), round(pos) + 1)
                        for pos in indexing_order[1:]
                    ]
                )
            )
        return vec

    def mean(self, axis):
        return vec3(
            np.mean(self.x, axis=axis),
            np.mean(self.y, axis=axis),
            np.mean(self.z, axis=axis),
        )

    def sum(self, axis):
        return vec3(
            np.sum(self.x, axis=axis),
            np.sum(self.y, axis=axis),
            np.sum(self.z, axis=axis),
        )

    def max(self):
        return vec3(max(self.x), max(self.y), max(self.z),)

    def min(self):
        return vec3(min(self.x), min(self.y), min(self.z),)

    def __eq__(self, other):
        return (self.x == other.x) & (self.y == other.y) & (self.z == other.z)

    def __lt__(self, scalar):
        """ less than is here defined for vec3 < y, y is a scalar """
        return (self.x < scalar) & (self.y < scalar) & (self.z < scalar)


def array_to_vec3(array):
    return vec3(array[0], array[1], array[2])


global rgb
rgb = vec3
