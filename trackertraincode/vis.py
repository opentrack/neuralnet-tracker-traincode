from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import torch
from matplotlib import pyplot
from matplotlib.widgets import Button
from typing import Tuple, Union, Optional

from trackertraincode.datasets.batch import Batch
from trackertraincode.datatransformation import _ensure_image_nhwc


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


def draw_points3d(img, pt3d, size=3, color = None, labels=False):
    assert pt3d.shape[-1] in (2,3)
    if color is None:
        color = (255,255,255)
    r,g,b = color
    for i, p in enumerate(pt3d[:,:2]):
        p = tuple(p.astype(int))
        if labels:
            cv2.putText(img, str(i), (p[0]+2,p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(img, p, size+1, (255,255,255), -1)
        cv2.circle(img, p, size, (r,g,b), -1)


def draw_roi(img, roi, color, linewidth):
    cv2.rectangle(img, (round(roi[0]),round(roi[1])), (round(roi[2]),round(roi[3])), color, linewidth)


def draw_pose(img, sample, color=None, linewidth=3):
    rot = sample['pose']
    x, y, s = sample['coord']
    draw_axis(img, rot, tdx = x, tdy = y, brgt=255, lw=linewidth, color=color)
    if s <= 0.:
        print (f"Error, head size {s} not positive!")
        print (sample)
    else:
        if color is None:
            color = (200,200,0)
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


def _draw_sample(img : np.ndarray, sample : Union[Batch,dict], labels : bool = True, color : Optional[tuple[int,int,int]] = None):
    linewidth = 2
    if 'seg_image' in sample:
        semseg = draw_semseg_class_indices(sample['seg_image'])
        img //= 2
        img += semseg // 2
    if 'pose' in sample and 'coord' in sample:
        draw_pose(img, sample, color, linewidth)
    if 'roi' in sample:
        #color = ((0,brightness,0) if sample['hasface']>0.5 else (brightness,0,0)) \
        #        if 'hasface' in sample else (brightness,0,brightness)
        roi = sample['roi']
        draw_roi(img, roi, color, linewidth)
    if 'hasface' in sample:
        maybe_draw_no_face_indication(img, sample, 255, linewidth)
    if 'pt3d_68' in sample:
        draw_points3d(img, sample['pt3d_68'], 
            linewidth-1, color, labels)
    if 'pt2d_68' in sample:
        draw_points3d(img, sample['pt2d_68'], 
            linewidth-1, color, labels)


def draw_prediction(sample_pred : Tuple[Batch,dict]):
    sample, pred = sample_pred
    img = np.ascontiguousarray(_with3channels_hwc(sample['image'].copy()))
    _draw_sample(img, sample, False, GT_COLOR)
    _draw_sample(img, pred, False, PRED_COLOR)
    return img


def draw_dataset_sample(sample : Batch, label=False):
    sample = dict(sample.items())
    img = np.ascontiguousarray(_with3channels_hwc(sample['image'].copy()))
    _draw_sample(img, sample, label, None)
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
