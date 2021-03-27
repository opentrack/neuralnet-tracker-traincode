from scipy.spatial.transform import Rotation
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
from matplotlib import pyplot
from matplotlib.widgets import Button

import utils


def draw_axis(img, rot, tdx=None, tdy=None, size = 100, brgt = 255, lw=3):
    if isinstance(rot, Rotation):
        rot = Rotation.as_matrix(rot)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # This projects the y-z plane onto the image.
    # Using the right hand rule, i.e. looking at my fingers
    # it is easy to recognize that I have to negate the
    # z component. The y component is negated because pixels
    # are counted from top down.
    rot = size*rot
    x1, x2, x3 = -rot[2,:] + tdx
    y1, y2, y3 = -rot[1,:] + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(brgt,0,0),lw)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,brgt,0),lw)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0,0,brgt),lw)

    return img


def draw_points3d(img, pt3d, labels=True, brightness=255):
    for i, p in enumerate(pt3d[:2,:].T):
        r = b = brightness
        p = tuple(p.astype(int))
        if labels:
            cv2.putText(img, str(i), (p[0]+2,p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(img, p, 2, (r,brightness,b), -1)


def _unnormalize_sample(img, sample):
    H, W, _ = img.shape
    result = {}
    if 'pose' in sample and 'coord' in sample:
        result['pose'] = utils.convert_to_rot(sample['pose'].cpu().numpy())
        result['coord'] = utils.unnormalize_coords((H,W), sample['coord'].cpu().numpy())
    if 'roi' in sample:
        result['roi'] = utils.unnormalize_boxes((H,W), sample['roi'].cpu().numpy())
    if 'hasface' in sample:
        result['hasface'] = sample['hasface'].cpu().numpy()
    if 'pt3d_68' in sample:
        result['pt3d_68'] = utils.unnormalize_points((H,W), sample['pt3d_68'].cpu().numpy())
    return result


def unnormalize_sample_to_numpy(sample, pred=None):
    # Not a batch?!
    assert 'image' in sample and len(sample['image'].shape)==3
    img = utils.unnormalize_images(sample['image'].cpu().numpy())
    # Return to conventional H x W X C format.
    # Also convert grayscale to RGB by simply repeating the channel.
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    img = np.ascontiguousarray(np.rollaxis(img, 0, 3))
    result = {
        'image' : img
    }
    result.update(_unnormalize_sample(img, sample))
    if pred is None:
        return result
    else:
        predresult = _unnormalize_sample(img, pred)
        return result, predresult


def _draw_sample(img, sample, is_prediction, labels=True):
    linewidth = 1 if is_prediction else 2
    brightness = 255 if is_prediction else 128
    if 'pose' in sample and 'coord' in sample:
        rot = sample['pose']
        x, y, s = sample['coord']
        draw_axis(img, rot, tdx = x, tdy = y, brgt=brightness, lw=linewidth)
        if s <= 0.:
            print (f"Error, head size {s} not positive!")
        else:
            cv2.circle(img, (int(x),int(y)), int(s), (brightness,brightness,0), linewidth)
    if 'roi' in sample:
        color = ((0,brightness,0) if sample['hasface']>0.5 else (brightness,0,0)) \
                if 'hasface' in sample else (brightness,0,brightness)
        roi = sample['roi']
        cv2.rectangle(img, (int(roi[0]),int(roi[1])), (int(roi[2]),int(roi[3])), color, linewidth)
    if 'hasface' in sample and sample['hasface']<0.5:
        color = (brightness,0,0)
        cv2.line(img, (0,0), (img.shape[1],img.shape[0]), color, linewidth)
        cv2.line(img, (0,img.shape[0]), (img.shape[1],0), color, linewidth)
    if 'pt3d_68' in sample:
        draw_points3d(img, sample['pt3d_68'], 
            labels, brightness)


def draw_prediction(ax, sample_pred):
    sample, pred = sample_pred
    img = sample['image'].copy()
    _draw_sample(img, sample, False, False)
    _draw_sample(img, pred, True, False)
    ax.imshow(img)


def draw_dataset_sample(ax, sample, label=True):
    img = sample['image'].copy()
    _draw_sample(img, sample, False, label)
    ax.imshow(img)


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
                drawfunc(ax, sample)
            ax.set_axis_off()
        if reset:
            it.reset()
        fig.canvas.draw()

    axnext = pyplot.axes([0.45, 0.01, 0.1, 0.05])
    button = Button(axnext,'Next')
    button.on_clicked(show_next_samples)

    show_next_samples()

    return fig, button