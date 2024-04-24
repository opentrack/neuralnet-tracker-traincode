from scipy.spatial.transform import Rotation
import numpy as np
import trimesh
import pyrender
from trackertraincode.facemodel.bfm import BFMModel


def _adjust_camera(camera_node : pyrender.Node, image_shape, background_plane_z_coord, scale):
    cam : pyrender.PerspectiveCamera = camera_node.camera
    h, w, _ = image_shape
    zdistance = 10000
    fov = 2.*np.arctan(0.5*(h)/(zdistance + background_plane_z_coord))
    cam.yfov=fov
    cam.znear = zdistance-scale*2
    cam.zfar = zdistance+scale*2
    campose = np.eye(4)
    campose[:3,3] = [ w//2, h//2, -zdistance  ]
    campose[:3,:3] = [
        [ 1, 0, 0 ],
        [ 0, 0, -1 ],
        [ 0, -1, 0 ]
    ]
    camera_node.matrix = campose


def _estimate_vertex_normals(vertices, tris):
    face_normals = trimesh.Trimesh(vertices, tris).face_normals
    new_normals = trimesh.geometry.mean_vertex_normals(len(vertices), tris, face_normals)
    assert new_normals.shape == vertices.shape, f"{new_normals.shape} vs {vertices.shape}"
    return new_normals


def _rotvec_between(a, b):
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    axis_x_sin = np.cross(a,b)
    cos_ = np.dot(a,b)
    if cos_ < -1.+1.e-6:
        return np.array([0.,np.pi,0.])
    if cos_ < 1.-1.e-6:
        return axis_x_sin/np.linalg.norm(axis_x_sin)*np.arccos(cos_)
    return np.zeros((3,))


def _direction_vector_to_pose_matrix(v):
    v = v / np.linalg.norm(v,keepdims=True)
    pose = np.eye(4)
    pose[:3,:3] = Rotation.from_rotvec(_rotvec_between(np.asarray([0., 0., -1.]),v)).as_matrix()
    return pose


class FaceRender(object):
    def __init__(self):
        self._bfm = BFMModel(40, 10)
        self._mat = pyrender.MetallicRoughnessMaterial(doubleSided=True, roughnessFactor=0.1, metallicFactor=0.0)
        vertices = self._bfm.scaled_vertices
        normals = _estimate_vertex_normals(self._bfm.scaled_vertices, self._bfm.tri)
        self._node = pyrender.Node(
            mesh=pyrender.Mesh(primitives = [pyrender.Primitive(positions = vertices, indices=self._bfm.tri, material=self._mat, normals=normals)]), 
            matrix=np.eye(4))
        self._scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])
        self._scene.add_node(self._node)
        self._light = pyrender.light.DirectionalLight(intensity=15.)
        self._scene.add_node(pyrender.Node(light = self._light, matrix=_direction_vector_to_pose_matrix([1.,0.,-10.])))
        self._camera_node = pyrender.Node(
            camera=pyrender.PerspectiveCamera(yfov=0.1, znear = 1., zfar = 10.),
            matrix=np.eye(4))
        _adjust_camera(self._camera_node, (640,640,None), 0., scale=240)
        self._scene.add_node(self._camera_node)
        self._renderer = pyrender.OffscreenRenderer(viewport_width=240, viewport_height=240)

    def set(self, xy, scale, rot, shapeparams, image_shape):
        '''Parameters must be given w.r.t. image space'''
        h, w = image_shape
        _adjust_camera(self._camera_node, (h,w,None), 0., scale=w)
        vertices = self._bfm.scaled_vertices + np.sum(self._bfm.scaled_bases * shapeparams[:,None,None], axis=0)
        normals = _estimate_vertex_normals(vertices, self._bfm.tri)
        self._node.mesh = pyrender.Mesh(primitives = [pyrender.Primitive(positions = vertices, indices=self._bfm.tri, material=self._mat, normals=normals)])
        matrix = np.eye(4)
        matrix[:3,:3] = scale * rot.as_matrix()
        matrix[:2,3] = xy
        self._node.matrix = matrix
        self._renderer.viewport_height = h
        self._renderer.viewport_width = w
        rendering, depth = self._renderer.render(self._scene)
        return rendering
