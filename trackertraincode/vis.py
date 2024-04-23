from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import torch
from matplotlib import pyplot
from matplotlib.widgets import Button
from typing import Tuple, Union, Optional

import trimesh
import pyrender

from trackertraincode.datasets.batch import Batch
from trackertraincode.datatransformation import _ensure_image_nhwc
from trackertraincode.facemodel.bfm import BFMModel

PRED_COLOR=(0,0,255)
GT_COLOR=(0,200,0)

def _with3channels_hwc(img : np.ndarray):
    assert len(img.shape) == 3
    img = _ensure_image_nhwc(img)
    if img.shape[-1]==1:
        img = np.tile(img,(1,1,3))
    return img


def draw_axis(img, rot, tdx=None, tdy=None, size = 100, brgt = 255, lw=3, color : Optional[tuple[int,int,int]] = None):
    if isinstance(rot, Rotation):
        rot = Rotation.as_matrix(rot)
    elif isinstance(rot, (np.ndarray, torch.Tensor)) and rot.shape==(4,):
        rot = Rotation.from_quat(rot).as_matrix()

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    rot = size*rot
    x1, x2, x3 = rot[0,:] + tdx
    y1, y2, y3 = rot[1,:] + tdy

    if color is None:
        xcolor = (brgt,0,0)
        ycolor = (0,brgt,0)
        zcolor = (0,0,brgt)
    else:
        r,g,b = color
        zcolor = ycolor = xcolor = (brgt*r//255,brgt*g//255,brgt*b//255)

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),xcolor,lw)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),ycolor,lw)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),zcolor,lw)

    return img


def draw_points3d(img, pt3d, labels=True, is_pred=False):
    assert pt3d.shape[-1] in (2,3)
    if is_pred:
        r,g,b = PRED_COLOR
    else:
        r,g,b = GT_COLOR
    for i, p in enumerate(pt3d[:,:2]):
        p = tuple(p.astype(int))
        if labels:
            cv2.putText(img, str(i), (p[0]+2,p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(img, p, 4, (255,255,255), -1)
        cv2.circle(img, p, 3, (r,g,b), -1)


def draw_roi(img, roi, color, linewidth):
    cv2.rectangle(img, (round(roi[0]),round(roi[1])), (round(roi[2]),round(roi[3])), color, linewidth)


def draw_pose(img, sample, is_prediction):
    rot = sample['pose']
    x, y, s = sample['coord']
    color = PRED_COLOR if is_prediction else GT_COLOR
    linewidth = 3
    draw_axis(img, rot, tdx = x, tdy = y, brgt=255, lw=linewidth, color=color)
    if s <= 0.:
        print (f"Error, head size {s} not positive!")
        print (sample)
    else:
        cv2.circle(img, (int(x),int(y)), int(s), color, linewidth)


def plot3dlandmarks(ax, keypts):
    xs, ys, zs = keypts.T
    ax.scatter(xs, ys, zs, s=3.)
    for i, p in enumerate(keypts):
        ax.text(p[0], p[1], p[2], s=str(i), size=9)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


def maybe_draw_no_face_indication(img, sample, brightness, linewidth):
    if sample['hasface']<0.5:
        color = (brightness,0,0)
        cv2.line(img, (0,0), (img.shape[1],img.shape[0]), color, linewidth)
        cv2.line(img, (0,img.shape[0]), (img.shape[1],0), color, linewidth)


_ibug_semseg_colors = np.asarray([
(  0, 0  , 0 ),
(255, 255, 0 ),
(139, 76 , 57 ),
(139, 54 , 38 ),
(  0, 205, 0 ),
(  0, 138, 0 ),
(154, 50 , 205 ),
(72 , 118, 255 ),
(255, 165, 0 ),
(  0, 0  , 139 ),
(255, 0  , 0 ),
], dtype=np.uint8)


def draw_semseg_class_indices(semseg : np.ndarray):
        H, W, C = semseg.shape
        assert C==1, f"bad shape {semseg.shape}"
        return _ibug_semseg_colors[semseg.ravel(),:].reshape((H, W, -1))


def draw_semseg_logits(semseg : np.ndarray):
    probs = np.exp(semseg)
    H, W, C = probs.shape
    colored = np.sum(_ibug_semseg_colors[None,None,:,:].astype(np.float32) * probs[...,None], axis=-2)
    colored = np.clip(colored, 0., 255.).astype(np.uint8)
    return colored


def _draw_sample(img : np.ndarray, sample : Union[Batch,dict], is_prediction : bool, labels : bool = True):
    linewidth = 3 #1 if is_prediction else 2
    #brightness = 255 if is_prediction else 128
    if 'seg_image' in sample:
        semseg = draw_semseg_class_indices(sample['seg_image'])
        img //= 2
        img += semseg // 2
    if 'pose' in sample and 'coord' in sample:
        draw_pose(img, sample, is_prediction)
    if 'roi' in sample:
        #color = ((0,brightness,0) if sample['hasface']>0.5 else (brightness,0,0)) \
        #        if 'hasface' in sample else (brightness,0,brightness)
        color = PRED_COLOR if is_prediction else GT_COLOR
        roi = sample['roi']
        draw_roi(img, roi, color, linewidth)
    if 'hasface' in sample:
        maybe_draw_no_face_indication(img, sample, 255, linewidth)
    if 'pt3d_68' in sample:
        draw_points3d(img, sample['pt3d_68'], 
            labels, is_pred=is_prediction)
    if 'pt2d_68' in sample:
        draw_points3d(img, sample['pt2d_68'], 
            labels, is_pred=is_prediction)


def draw_prediction(sample_pred : Tuple[Batch,dict]):
    sample, pred = sample_pred
    img = np.ascontiguousarray(_with3channels_hwc(sample['image'].copy()))
    _draw_sample(img, sample, False, False)
    _draw_sample(img, pred, True, False)
    return img


def draw_dataset_sample(sample : Batch, label=False):
    sample = dict(sample.items())
    img = np.ascontiguousarray(_with3channels_hwc(sample['image'].copy()))
    _draw_sample(img, sample, False, label)
    return img


def matplotlib_plot_iterable(iterable, drawfunc, rows=3, cols=3, figsize=(10,10)):
    '''
        Generates a tile grid of plots showing items from 'iterable'.
        The 'drawfunc' takes an item and an axes and is responsible for
        the actual drawing.
        There is also a button to show advance to the next items.
        
        Returns figure and button.
        
        Note: use %matplotlib notebook in order for the button to work.
    '''
    # Make tile grid of blank images
    fig, axes = pyplot.subplots(rows, cols, figsize=figsize)
    blank = np.zeros((1,1,3), np.uint8)    
    for ax in axes.ravel():
        ax.set_axis_off()
        ax.imshow(blank)
        ax.set_title(" ")
    pyplot.tight_layout()

    class ResetableIter(object):
        # Used to capture the iterable
        # and allow iteration over it to be restarted 
        # in show_next_samples, i.e. to allow reassignment
        # to 'it' within show_next_samples.
        def __init__(self, ds):
            self.ds = ds
            self.it = iter(ds)
        def next(self):
            return next(self.it)
        def reset(self):
            self.it = iter(self.ds)
    
    it = ResetableIter(iterable)

    def show_next_samples(*args):
        # Iterate over subplots and fill them with sample visualizations.
        # When done fill remaining plots with blank image.
        # Next round start from the beginning of the dataset.
        reset = False
        for ax in axes.ravel():
            try:
                sample = it.next()
            except StopIteration:
                ax.set_title(" ")
                ax.clear()
                ax.imshow(blank)
                reset = True
            else:
                img = drawfunc(sample)
                ax.imshow(img)
            ax.set_axis_off()
        if reset:
            it.reset()
        fig.canvas.draw()

    axnext = pyplot.axes([0.45, 0.01, 0.1, 0.05])
    button = Button(axnext,'Next')
    button.on_clicked(show_next_samples)

    show_next_samples()

    return fig, button


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

# TODO: refactor -> vis 3d module
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
