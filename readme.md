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

* Kaggle YT Face videos
https://www.kaggle.com/selfishgene/youtube-faces-with-facial-keypoints
3d keypoints fitted with neural network. Some manual cleaning. Noisy, but still helps, I guess. No pose data unfortunately, so can only use keypoints as training target.

* WIDER FACE
http://shuoyang1213.me/WIDERFACE/index.html
For face detection.

Usage
=====

There is not proper packaging yet. Therefore the project dir should be added to the python search path. The Anaconda environment
and package manager is highly recommended. Assuming you have that, and work under Linux, you can get started by
```bash
cd dir/with/tracker-traincode
```
```bash
conda activate ml   # Activate your anaconda env
export PYTHONPATH=`pwd`  # Set pythonpath so search in current work dir
export DATADIR=dir/with/data/files # Scripts look there by default
```

Regarding the datasets. Download them. Then run the conversion scripts. A script like the following is recommended.
```bash
#!/bin/bash
# Use -n <number> to limit the number of data points for faster development and testing
# python scripts/dsbiwi_processing.py      $@ $DATADIR/biwi.zip $DATADIR/biwi.h5 # Don't really need this ...
python scripts/dsaflw2k_processing.py    $@ $DATADIR/AFLW2000-3D.zip $DATADIR/aflw2k.h5
python scripts/ds300wlp_processing.py    $@ $DATADIR/300W-LP.zip $DATADIR/300wlp.h5
python scripts/dsytfaces_processing.py   $@ $DATADIR/YTFaces $DATADIR/ytfaces.h5
python scripts/dswiderface_processing.py $@ $DATADIR/wider_faces $DATADIR/widerfacessingle.h5
```

Check the data with the help of the notebook `DataVisualization.ipynb`.

Run training in the notebooks `TrainLocalizer.ipynb` and `TrainKeypoints.ipynb`.

The result can be inspected with `LocalizerEvaluation.ipynb` and `PoseNetworkEvaluation.ipynb`.

Afterwards the networks must be converted to the ONNX format. ONNX is Microsofts storage format which happens to be supported by a relatively lightweight runtime of the same name, allowing inference on the CPU. To carry out this conversion there is `export_model_onnx.py` in the scripts folder together with all the other stuff.

Dependencies
============
```
Python, PyTorch, Jupyter, OpenCV, SciPy, H5py, Progressbar2, ONNX
```
Miscellaneous
=============

Comparison with literature references
-------------------------------------

It is common in the head-pose estimation literature to compare yaw, pitch and roll errors  - more precisely the mean absolute error (MAE), which is simply the average of `|xhat - x|` taken over the test set, where `xhat` is the angle prediction and `x` the ground truth value respectively.

There are two recent works which can be taken for reference:
[1] Albiero et al. (2021) "img2pose: Face Alignment and Detection via 6DoF, Face Pose Estimation"
[2] Hsu et al. (2018) "Quatnet: Quaternion-based head pose estimation with multiregression loss"

With results for the popular AFLW 2000 3D benchmark:

| Method            | Yaw   | Pitch | Roll  | Average |
|-------------------|-------|-------|-------|---------|
| QuatNet           | 3.973 | 5.615 | 3.920 | 4.503   |
| img2pose          | 3.426 | **5.034** | **3.278** | **3.913** |
| **NN-Tracker (this work)** | **3.375** | 5.222 | 3.560 | 4.052 |

The values for QuatNet and img2pose are taken from [1]. The values for NN-Tracker were generated with the training code for first release with [opentrack 2021.1.2](https://github.com/opentrack/opentrack/releases/tag/opentrack-2021.1.2). For the comparison, I retrained a network, not training on AFLW 2000 3D since I was using it as test set. See [notebook in feature branch](https://github.com/opentrack/neuralnet-tracker-traincode/blob/proper-measurement/scripts/TrainKeypoints.ipynb).

The authors of 3DDFA_V2 did not publish angle errors. A comparison of landmark position errors is still to be done ...

Head coordinate frame
---------------------
X is forward from the viewpoint of the faces. Y is up. Z is right. Granted, my choice is a bit awkward ... but tbh I'm happy I got it working at all.

When viewed from the front, the face has identity rotation, meaning its local axes are aligned with the world axes.

Camera space is different. Here X is right, Y is down and Z is out of the screen (I think). So, to get from world space to camera or image space there is an additional transformation.

I should probably fix it so it's all the same.

OpenCV Performance
------------------

The OpenCV from Conda Forge run extremely slowly. Turns out it is better if you force it to use only one thread. Hence.
```
import cv2
cv2.setNumThreads(1)
```

Before I tried that I made the ugly hack with image augmentation on the GPU using the PostprocessingDataLoader in datatransformation.py with which I can run stuff on the gpu after a batch was assembled. Running cuda code on the worker processes is unfortunately not possible so an extra processing step has to take place.

See also
https://github.com/ContinuumIO/anaconda-issues/issues/10041
https://github.com/opencv/opencv/issues/11107#issuecomment-393475735

Licensing
=========

This software, that is everything not covered by other licenses is published under the ISC license.

Copyright 2021 Michael Welter

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Deformable Face Model
---------------------

The modified BFM2009 face model in datasets/bfm_noneck_v3.pkl is only for academic use. For commercial use, you need to apply for the commercial license, some refs are below:

[3] https://faces.dmi.unibas.ch/bfm/?nav=1-0&id=basel_face_model

[4] https://faces.dmi.unibas.ch/bfm/bfm2019.html

[5] P. Paysan et al. (2009) "A 3D Face Model for Pose and Illumination Invariant Face Recognition"

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

AFLW2000-3d / AFLW
------------------

AFLW2000-3d is derived from AFLW (https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/). The authors of AFLW2000-3d don't provide a license agreement. Hence, in the following the license of AFLW is reproduced:

License Agreement

By downloading the database you agree to the following restrictions:

* The AFLW database is available for non-commercial research purposes only.
* The AFLW database includes images obtained from FlickR which are not property of Graz University of Technology. Graz University of Technology is not responsible for the content nor the * meaning of these images. Any use of the images must be negociated with the respective picture owners, according to the Yahoo terms of use. In particular, you agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
* You agree not to further copy, publish or distribute any portion of the AFLW database. Except, for internal use at a single site within the same organization it is allowed to make copies of the database.
* All submitted papers or any publicly available text using the AFLW database must cite our paper
* The organization represented by you will be listed as users of the AFLW database.

300W-LP / 300W
--------------

300W-LP is derived from 300W (https://ibug.doc.ic.ac.uk/resources/300-W/). 300W explicitly prohibits commercial use as stated on the project website. Citation:
> The data are provided for research purposes only. Commercial use (i.e., use in training commercial algorithms) is not allowed.

YouTube Faces With Facial Keypoints
-----------------------------------

Published under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) which is a permissive license. This data set is derived from [YouTube Faces](https://www.cs.tau.ac.il/~wolf/ytfaces/) for which I did not find terms of use.

WIDER FACE
----------

I did not find terms of use for this dataset specifically. However it is based on [WIDER](http://yjxiong.me/event_recog/WIDER/), where you have to agree to using the dataset for research purposes only before downloading.