from scipy.spatial.transform import Rotation
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
from matplotlib import pyplot
from matplotlib.widgets import Button
import enum

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


def draw_points3d(img, pt3d, labels=True, brightness=255, color=None):
    if color is not None:
        r, g, b = color
    else:
        g = brightness
        b = r = brightness//2
    for i, p in enumerate(pt3d[:2,:].T):
        p = tuple(p.astype(int))
        if labels:
            cv2.putText(img, str(i), (p[0]+2,p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(img, p, 1, (r,g,b), -1)


def draw_roi(img, roi, color, linewidth):
    cv2.rectangle(img, (round(roi[0]),round(roi[1])), (round(roi[2]),round(roi[3])), color, linewidth)


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
        draw_roi(img, roi, color, linewidth)
    if 'roi_head' in sample:
        color = (brightness,brightness,brightness)
        draw_roi(img, sample['roi_head'], color, linewidth)
    if 'hasface' in sample and sample['hasface']<0.5:
        color = (brightness,0,0)
        cv2.line(img, (0,0), (img.shape[1],img.shape[0]), color, linewidth)
        cv2.line(img, (0,img.shape[0]), (img.shape[1],0), color, linewidth)
    if 'pt3d_68' in sample:
        draw_points3d(img, sample['pt3d_68'], 
            labels, brightness)


def draw_prediction(sample_pred):
    sample, pred = sample_pred
    img = sample['image'].copy()
    _draw_sample(img, sample, False, False)
    _draw_sample(img, pred, True, False)
    return img


def draw_dataset_sample(sample, label=True):
    img = sample['image'].copy()
    _draw_sample(img, sample, False, label)
    return img


def _compress(img):
    _, img = cv2.imencode('.PNG', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img.tobytes()


def _generate_items(xrange, yrange, generate_sample):
    return sum([ 
        [ generate_sample(i,j) for i in xrange
        ] for j in yrange ]
    , [])


def _assemble_collage(images, cols):
    shape = np.amax(np.asarray([ img.shape for img in images ]), axis=0)
    for i, img in enumerate(images):
        if img.shape != tuple(shape):
            tmp = np.zeros(shape, dtype=np.uint8)
            tmp[:img.shape[0],:img.shape[1],:] = img
            images[i] = tmp
    images = [ *utils.iter_batched(images, cols) ]
    images = np.vstack([ np.hstack(row) for row in images ])
    return images


class Backend(enum.Enum):
    JUPYTER = 1
    WINDOW = 2


def display_image_panel_simple(generate_sample, rows : int, cols : int, title='Image Panel', oneway=False, stride=1):
    import PySimpleGUI as sg

    offset_col = 0

    def regenerate_collage():
        return _compress(_assemble_collage(
            _generate_items(range(offset_col, cols+offset_col),range(rows), generate_sample),
            cols))

    buttons = [ sg.Button("Next") ]
    if not oneway:
        buttons = [ sg.Button("Prev") ] + buttons
    layout = [
        [ sg.Image(key="Image", data=regenerate_collage()) ],
        buttons
    ]
    # Create the window
    window = sg.Window(title, layout)
    # Create an event loop
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == 'Prev' or event == 'Next':
            offset_col += stride if event == 'Next' else -stride
            window['Image'].update(data=regenerate_collage())

    window.close()


def display_image_panel_jupyter(generate_sample, rows : int, cols : int, oneway=False, stride=1):
    from ipywidgets import Layout, Box, VBox, HBox, Image, Button

    offset_col = 0

    def regenerate_collage():
        return _assemble_collage(
            _generate_items(range(offset_col, cols+offset_col),range(rows), generate_sample),
            cols)

    initial = regenerate_collage()

    img_widget = Image(
        format='raw',
        value = _compress(initial),
        width=initial.shape[1],
        height=initial.shape[0],
    )

    next = Button(description='Next', layout=Layout(width='auto'))
    def on_next(*args):
        nonlocal offset_col
        offset_col += stride
        img_widget.value = _compress(regenerate_collage())
    next.on_click(on_next)

    if not oneway:
        prev = Button(description='Prev', layout=Layout(width='auto'))
        def on_prev(*args):
            nonlocal offset_col
            offset_col -= stride
            img_widget.value = _compress(regenerate_collage())
        prev.on_click(on_prev)

    buttons = [ next ]
    if not oneway:
        buttons = [ prev ] + buttons

    box = VBox([HBox(buttons), img_widget])
    return box


def display_image_panel(backend : Backend, *args, **kwargs):
    if backend == Backend.JUPYTER:
        return display_image_panel_jupyter(*args, **kwargs)
    else:
        return display_image_panel_simple(*args, **kwargs)


def display_image_panel_iterable(backend : Backend, iterable, drawfunc, rows : int = 3, cols : int = 3, **kwargs):
    '''
        Generates a tile grid of plots showing items from 'iterable'.
        The 'drawfunc' takes an item and an axes and is responsible for
        the actual drawing.
        There is also a button to show advance to the next items.
        
        Returns figure and button.
        
        Note: use %matplotlib notebook in order for the button to work.
    '''
    it = iter(iterable)
    def generate_sample(col, row):
        nonlocal it
        try:
            img = drawfunc(next(it))
        except StopIteration:
            img = np.zeros((1,1,3), dtype=np.uint8)
        return img
    return display_image_panel(backend, generate_sample, rows, cols, oneway=True, stride=cols, **kwargs)


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