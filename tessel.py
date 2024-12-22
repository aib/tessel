from collections import defaultdict
from itertools import pairwise
import math
import random

import cairo
import PIL.Image
import numpy
import scipy

from mesh import *

class Parameters:
	mesh_size = (32, 32)
	lith_thickness = (0.24, 4.8)
	frame_points = 128
	frame_padding = 1.0
params = Parameters()

def lerp(a, b, t):
	return (1.0 - t) * a + t * b

def iter_circle(iterable):
	iterator = iter(iterable)
	first = next(iterator)
	yield first
	yield from iterator
	yield first

def circular(iterable):
	return pairwise(iter_circle(iterable))

def load_image(filename):
	with PIL.Image.open(filename) as im:
		im = im.convert('L')
		imarr = numpy.asarray(im)[::-1].swapaxes(0, 1)
		imarr = imarr / 255.0

	width = imarr.shape[0]
	height = imarr.shape[1]

	xs = numpy.linspace(0, width, width)
	ys = numpy.linspace(0, height, height)
	interpolator = scipy.interpolate.RegularGridInterpolator((xs, ys), imarr, method='linear')

	return (interpolator, (width, height))

def get_vertex_colors(interpolator, image_size, vertices):
	vertices_scaled = vertices * [image_size[0] / params.mesh_size[0], image_size[1] / params.mesh_size[1]]
	vertices_scaled_clipped = numpy.empty_like(vertices_scaled)
	vertices_scaled_clipped[:,0] = numpy.clip(vertices_scaled[:,0], 0, image_size[0])
	vertices_scaled_clipped[:,1] = numpy.clip(vertices_scaled[:,1], 0, image_size[1])
	return interpolator(vertices_scaled_clipped)

def find_bounding_circle(vertices, center):
	deltas = vertices - center
	dists = (deltas * deltas).sum(axis=-1)
	dmax = math.sqrt(dists.max())
	return (dmax, center)

def main():
	interp, (image_width, image_height) = load_image('PIA00405.tif')

	mesh = rectangle_equimesh(params.mesh_size[0], params.mesh_size[1])
	vertex_colors = get_vertex_colors(interp, (image_width, image_height), mesh.vertices)
	vert3d = numpy.concatenate((mesh.vertices, vertex_colors.reshape(-1, 1)), axis=1)
	mesh.vertices = numpy.concatenate((mesh.vertices, vertex_colors.reshape(-1, 1)), axis=1)

	good_faces = (mesh.vertices[mesh.faces][:, :, 2] > 0).any(axis=1)
	mesh.faces = mesh.faces[good_faces]

	build_3d(mesh)
	with open("/tmp/out.obj", 'w') as f:
		mesh.save_obj(f)

	image_size = numpy.array([image_width, image_height])
	svg_border = image_size * 0.1
	surface_size = image_size + svg_border
	with cairo.SVGSurface("/tmp/tess/ex13.svg", surface_size[0], surface_size[1]) as surface:
		ctx = cairo.Context(surface)
		ctx.translate(0, surface_size[1])
		ctx.scale(1.0, -1.0)

		ctx.translate(svg_border[0] / 2, svg_border[1] / 2)
		ctx.scale(image_width / params.mesh_size[0], image_height / params.mesh_size[1])

		for i, tri in enumerate(mesh.polygons):
			col = tri[:,2].sum() / 3
			ctx.set_source_rgba(col, col, col, 1)
			ctx.move_to(tri[0][0], tri[0][1])
			ctx.line_to(tri[1][0], tri[1][1])
			ctx.line_to(tri[2][0], tri[2][1])
			ctx.line_to(tri[0][0], tri[0][1])
			ctx.fill()

def build_3d(mesh, build_base=True):
	# Add frame
	radius, center = find_bounding_circle(mesh.polygons.reshape(-1, 3)[:,0:2], (params.mesh_size[0] / 2, params.mesh_size[1] / 2))
	padded_radius = radius + params.frame_padding

	angles = numpy.linspace(0, math.tau, params.frame_points, endpoint=False)
	perimeter_vertices = numpy.stack([
		center[0] + numpy.sin(-angles) * padded_radius,
		center[1] + numpy.cos(-angles) * padded_radius,
		numpy.zeros_like(angles),
	], axis=1)
	perimeter = mesh.add_vertices(perimeter_vertices)
	hull = mesh.outer_hull_2d

	deltas = mesh.vertices[hull, None, 0:2] - perimeter_vertices[None, :, 0:2]
	dists_sq = numpy.square(deltas).sum(axis=2)
	#closest_hull = hull[numpy.argmin(dists_sq, axis=0)]
	closest_perimeter = perimeter[numpy.argmin(dists_sq, axis=1)]

	perimeter_connections = defaultdict(lambda: [])
	for (h0, p0), (h1, _) in circular(zip(mesh.outer_hull_2d, closest_perimeter)):
		mesh.add_faces([(h0, p0, h1)])
		perimeter_connections[p0].append(h0)
		perimeter_connections[p0].append(h1)

	# Get CW-most/CCW-most perimeter connections
	hull_list = list(hull)
	for p, hs in perimeter_connections.items():
		hull_indices = numpy.array(list(map(lambda h: hull_list.index(h), hs)))
		# Normalize to +/- len/2 around one of the values
		hi_compare = (hull_indices - hull_indices.min() + len(hull) // 2) % len(hull)
		himin, himax = hi_compare.argsort()[[0, -1]]
		perimeter_connections[p] = (hull[hull_indices[himin]], hull[hull_indices[himax]])

	# Make sure first point has connections
	perimeter_rotated = perimeter
	while len(perimeter_connections[perimeter_rotated[0]]) == 0:
		perimeter_rotated = numpy.roll(perimeter_rotated, -1)

	prev_conn = None
	for p0, p1 in circular(perimeter_rotated):
		cs0 = perimeter_connections[p0]
		if len(cs0) > 0:
			prev_conn = cs0[1]
		mesh.add_faces([(p0, p1, prev_conn)])

	# Adjust height
	mesh.vertices[:,2] = lerp(params.lith_thickness[1], params.lith_thickness[0], mesh.vertices[:,2])

	if build_base:
		base_vertices = perimeter_vertices.copy()
		base_vertices[:,2] = 0
		base = mesh.add_vertices(base_vertices)
		base_center = mesh.add_vertices([(params.mesh_size[0] / 2, params.mesh_size[1] / 2, 0)])[0]

		for (b0, p0), (b1, p1) in circular(zip(base, perimeter)):
			mesh.add_faces([(b0, b1, p0), (p0, b1, p1)])

		for b0, b1 in circular(base):
			mesh.add_faces([(b1, b0, base_center)])

if __name__ == '__main__':
	main()
