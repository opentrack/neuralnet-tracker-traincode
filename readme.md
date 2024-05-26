Intro
=====

This project contains the code to train the neural nets for my opentrack head tracker. The home page of the opentrack project is https://github.com/opentrack/opentrack.

There are two parts. A localizer network which computes a bounding box around the users head in the webcam video feed, typically.

And the second part is the actual pose network which looks at a region of interest around the head, i.e. the previously computed bounding box, and outputs the following:
* a quaternion representing the orientation
* x, y coordinates in screen space,
* head radius in screen space,
* facial key points, which are used as additional training objective,
* a new bounding box for the head. Currently is trained to return the bounding box of the key points.

It is pretty straight forward, except maybe the key points. I loosly follow the approach from https://github.com/cleardusk/3DDFA_V2, where key points are literally a few key vertices taken from a deformable face model. The convolutional backbone actually outputs parameters for the deformable model. In computer graphics this is sometimes called "blend shapes" since some deformation vectors are superimposed by linear combination.

Regarding localization, I use the Wider Face dataset which is for general face detection. But since my network only supports to find one face, I do execessive processing to generate pairs of images with exactly one and without face, respectively. Detection is hard and I wanted to mess around with my own networks.

Datasets
========

Several datasets are used. All of which are preprocessed and the result stored in h5 files.

* AFLW2000-3d & 300W-LP
http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
Generated with 3D Dense Face Alignment (3DDFA) to fit a morphable 3Dmodel to 2D input images and provide accurate head poses as ground-truth. 300W-LP additionally generates synthetic views, greatly expanding the number of images.

* WIDER FACE
http://shuoyang1213.me/WIDERFACE/index.html
For face detection.

* WFLW large pose augmented
TODO

* LAPA large pose augmented
TODO

Usage
=====

TODO: pip installable

Data preparation
----------------

Regarding the datasets. Download them. Then run the conversion scripts.

```bash
# Use -n <number> to limit the number of data points for faster development and testing
# For training the localizer network
python scripts/dswiderface_processing.py $@ $DATADIR/wider_faces $DATADIR/widerfacessingle.h5
# For training the pose estimator
python scripts/dsaflw2k_processing.py    $@ $DATADIR/AFLW2000-3D.zip $DATADIR/aflw2k.h5
python scripts/ds300wlp_processing.py    $@ $DATADIR/300W-LP.zip $DATADIR/300wlp.h5
```
TODO finish

Check the data with the help of the notebook `DataVisualization.ipynb`.

Training the localizer network
------------------------------

Run training in the notebooks `TrainLocalizer.ipynb`.
The result can be inspected with `LocalizerEvaluation.ipynb`

Afterwards the networks must be converted to the ONNX format. To carry out this conversion there
 is `export_model_onnx.py` in the scripts folder together with all the other stuff.

Training the pose estimation network
------------------------------------

TODO: `scripts/train_poseestimator.py`

```bash
python scripts/train_poseestimator.py --lr 1.e-3 --epochs 1500 --ds "repro_300_wlp+synface+lapa_megaface_lp+wflw_lp" --auglevel 2 \
    --save-plot train.pdf \
    --with-swa \
    --backbone mobilenetv1
```


Dependencies
============
```
Python, PyTorch, Jupyter, OpenCV, SciPy, H5py, tqdm, ONNX
```
Miscellaneous
=============

Head coordinate frame
---------------------

X is right, Y is down and Z is into the screen.


Licensing
=========

This software, that is everything not covered by other licenses is published under the ISC license.

Copyright 2021 Michael Welter

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Deformable Face Model
---------------------

The modified BFM2009 face model in datasets/bfm_noneck_v3.pkl is only for academic use. For commercial use, you need to apply for the commercial license, some refs are below:

[1] https://faces.dmi.unibas.ch/bfm/?nav=1-0&id=basel_face_model

[2] https://faces.dmi.unibas.ch/bfm/bfm2019.html

[3] P. Paysan et al. (2009) "A 3D Face Model for Pose and Illumination Invariant Face Recognition"

3DDFA V2
--------
(https://github.com/cleardusk/3DDFA_V2)

MIT License

Copyright (c) 2017 Max deGroot, Ellis Brown
Copyright (c) 2019 Zisian Wong, Shifeng Zhang
Copyright (c) 2020 Jianzhu Guo, in Center for Biometrics and Security Research (CBSR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.