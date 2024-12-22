from dataclasses import dataclass
import math

import numpy

@dataclass
class Mesh:
	def __init__(self, vertices, faces):
		self._vertices = vertices
		self._faces = faces
		self._hull2d = None
		self._outer_hull = None

	@property
	def vertices(self):
		return self._vertices

	@vertices.setter
	def vertices(self, vertices):
		self._vertices = vertices

	@property
	def faces(self):
		return self._faces

	@faces.setter
	def faces(self, faces):
		self._faces = faces

	@property
	def polygons(self):
		return self._vertices[self._faces]

	@property
	def convex_hull_2d(self):
		import scipy
		if self._hull2d is None:
			self._hull2d = scipy.spatial.ConvexHull(self._vertices[:,0:2]).vertices

		return self._hull2d

	@property
	def outer_hull_2d(self):
		if self._outer_hull is None:
			vert2d = self._vertices[:,0:2]
			edges = numpy.stack(
				(self._faces, numpy.roll(self._faces, -1, axis=1)),
				axis=2
			).reshape(-1, 2)
#			edges.sort()
#			edges = numpy.unique(edges, axis=0)

			mesh_vi = numpy.unique(edges.flatten())
			vi_vertices = vert2d[mesh_vi]

			vi = mesh_vi[vi_vertices[:,0].argmin()]
			last_angle = 3 * math.tau / 4

			hull = [vi]

			for _ in range(len(edges)):
				connections = edges[edges[:,0] == vi][:,1]

				deltas = vert2d[connections] - vert2d[vi]
				angles = numpy.atan2(deltas[:,1], deltas[:,0])
				choice_angles = -(angles - last_angle + math.pi) % math.tau

				chosen = choice_angles.argmax()
				vi = connections[chosen]
				last_angle = angles[chosen]

				if hull[0] == vi:
					break
				else:
					hull.append(vi)

			self._outer_hull = numpy.array(hull, dtype=self._faces.dtype)

		return self._outer_hull

	def clean(self):
		vertices = []
		remap = {}

		for vi in self._faces.flat:
			if vi not in remap:
				vertices.append(self._vertices[vi])
				remap[vi] = len(vertices) - 1

		return Mesh(numpy.asarray(vertices), numpy.vectorize(remap.get)(self._faces))

	def add_vertices(self, vertices):
		new_vertices = numpy.append(self._vertices, vertices, axis=0)
		indices = numpy.array(range(self._vertices.shape[0], new_vertices.shape[0]))
		self._vertices = new_vertices
		return indices

	def add_faces(self, faces):
		self._faces = numpy.append(self._faces, faces, axis=0)

	def save_obj(self, file):
		vdim = self._vertices.shape[1]
		if vdim == 2:
			for v in self._vertices:
				file.write(f'v {v[0]} {v[1]} 0.0\n')
		elif vdim == 3:
			for v in self._vertices:
				file.write(f'v {v[0]} {v[1]} {v[2]}\n')
		else:
			raise ValueError(f"Don't know how to write vertices with dimension {vdim}")

		for f in self._faces + 1:
			file.write(f'f {" ".join(map(str, f))}\n')

def rectangle_trimesh(width, height, dtype=None):
	ww = width + 1
	hh = height + 1
	vertices = numpy.indices((ww, hh), dtype=dtype).transpose(2, 1, 0).reshape(-1, 2)

	row_indices = numpy.arange(width)
	faces_arr = []

	for y in range(height):
		ri = row_indices + y * ww
		row_faces = numpy.stack((
			ri, ri + 1, ri + ww,
			ri + ww, ri + 1, ri + ww + 1,
		), axis=1)
		faces_arr.append(row_faces)

	faces = numpy.asarray(faces_arr).reshape(-1, 3)
	return Mesh(vertices, faces)

def rectangle_equimesh(width, height, meshmode='square_fit', dtype=None):
	if meshmode == 'tiling':
		ww = width + 2
		hh = height + 1
		row_indices1 = numpy.arange(ww - 1)
		row_indices2 = numpy.arange(ww - 1)
	elif meshmode == 'square_fit':
		ww = width + 2
		hh = height + 1
		row_indices1 = numpy.arange(ww - 1)
		row_indices2 = numpy.arange(ww - 2)
	else:
		raise ValueError("Invalid 'meshmode' argument")

	vertices = numpy.indices((ww, hh), dtype=dtype).transpose(2, 1, 0)
	vertices[1::2,:] -= [0.5, 0.0]
	vertices = vertices.reshape(-1, 2)

	faces = numpy.zeros((0, 3), dtype=row_indices1.dtype)

	for y in range(height):
		ri1 = row_indices1 + y * ww
		ri2 = row_indices2 + y * ww
		if y % 2 == 0:
			row_faces1 = numpy.stack((ri1 + ww, ri1, ri1 + ww + 1), axis=1)
			row_faces2 = numpy.stack((ri2 + ww + 1, ri2, ri2 + 1), axis=1)
			faces = numpy.append(faces, row_faces1, axis=0)
			faces = numpy.append(faces, row_faces2, axis=0)
		else:
			row_faces1 = numpy.stack((ri1, ri1 + 1, ri1 + ww), axis=1)
			row_faces2 = numpy.stack((ri2 + ww, ri2 + 1, ri2 + ww + 1), axis=1)
			faces = numpy.append(faces, row_faces1, axis=0)
			faces = numpy.append(faces, row_faces2, axis=0)

	mesh = Mesh(vertices, faces)
	return mesh
