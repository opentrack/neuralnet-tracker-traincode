Opentrack Neuralnet-Tracker Train Code
======================================

Intro
-----

This project contains the code to train the neural nets for [Opentrack](https://github.com/opentrack/opentrack)'s neuralnet-tracker. This tracking component performs localization and 6d pose estimation of the user's face in Opentrack's webcam feed.

There are two networks. There is one for localization which computes a bounding box around the users head, and a second one which performs the pose prediction. The latter looks at a region of interest around the head, i.e. using the previously computed bounding box, and outputs the following:

* a quaternion representing the orientation,
* x, y coordinates in screen space,
* head radius in screen space,
* facial key points, which are used as auxiliary training objective,
* a new bounding box. This enables tracking without the localization component.

It is straight forward, except maybe the key points. I loosely follow the approach from https://github.com/cleardusk/3DDFA_V2. Therein key points, a.k.a. landmarks, are taken from a deformable 3d face model. The convolutional backbone outputs blend weights for this face model.

Regarding localization, I train on the WIDER FACE dataset which is for general face detection. For Opentrack, localization of a single face is sufficient, so no full detection is done. The data set is processed such that a balanced set of positive and negative examples is created.

[Link to demo video](doc/opentrack-neuralnet-tracker.mp4).

The video demonstrates usability in practice, performance in relatively low light conditions, and stability under rapidly changing illumination. Throughout the video, the virtual camera zooms in and resets abruptly a few seconds later. This is due to user inputs and intended. Note also that the head pose passes through a low-pass filter before it is transferred to the camera.

Comparison with state of the art
--------------------------------

It is common in the head-pose estimation literature to compare yaw, pitch and roll errors - more precisely the mean absolute error (MAE), which is simply the average of `|xhat - x|` taken over the test set, where `xhat` is the angle prediction and `x` the ground truth value respectively.

The site paperswithcode.com provides a good [overview of the current state of the art](https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000) regarding the popular AFLW 2000 3D pose-estimation benchmark. In short, the neuralnet-tracker is better than the 2018 work "Quatnet". Later work beats it on this benchmark.

Below is a compilation of quantitative results from relevant publications:

* *[1] Hsu et al. (2018) "Quatnet: Quaternion-based head pose estimation with multiregression loss"*
* *[2] Valle et al. (2021) "Multi-Task Head Pose Estimation in-the-Wild"*
* *[3] Wu et al. (2021) "Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry"*
* *[4] Guo et al. (2020) "Towards Fast, Accurate and Stable 3D Dense Face Alignment"*

The following table shows the MAE in degrees:

| Method            | Yaw   | Pitch | Roll  | Average |
|-------------------|-------|-------|-------|---------|
| QuatNet [1]       | 3.973 | 5.615 | 3.920 | 4.503   |
| (3DDFA_V2 [4] \*\*\*) | **3.183** | 5.227 | 3.468 | 3.959   |
| Valle et al. [2]  | 3.34  | 4.69  | 3.48  | 3.83    |
| Wu et al. [3]     | 3.42  | **4.09**  | **2.55**  | **3.35** |
| **NN-Tracker \*** | 3.373 | 5.206 | 3.545 | 4.041 |
| **NN-Tracker \*\*** | 3.370 | 5.761 | 3.607 | 4.246 |

\* Inputs cropped to ground-truth face bounding boxes. This is what the network has been trained on.

\*\* Inputs cropped to centers. Center cropping has been reported in [2] but I found no details. So I just picked a fixed percentage that looked good. I did *not* optimize the cropped section for best results!

\*\*\* According to my own measurement. Cropping to bounding boxes of ground truth landmark annotations. See [3DDFA_V2 evaluation notebook](https://github.com/DaWelter/3DDFA_V2/blob/master/AFLW20003dEvaluation.ipynb).

3DDFA_V2 relies on bounding boxes from the FaceBoxes detector. This detector failed on a few images. When the input boxes are taken from the detector, excluding failures, the performance dropped to an average MAE of 4.010°.

For this comparison, I retrained a network, not training on AFLW 2000 3D since I was using it as test set. See [training notebook in feature branch](https://github.com/opentrack/neuralnet-tracker-traincode/blob/proper-measurement/scripts/TrainKeypoints.ipynb) and [evaluation notebook](https://github.com/opentrack/neuralnet-tracker-traincode/blob/proper-measurement/scripts/AFLW20003dEvaluation.ipynb)

Datasets
--------

Training is performed on the following data sets. They are pre-processed into hdf5 files, applying additional transformations such as coordinate system changes, cropping and rescaling.

* AFLW2000-3d & 300W-LP
http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
These datasets provide face images labeled with 6d poses and parameters for said deformable face model. 300W-LP additionally provides synthetic views, greatly expanding the number of images.

* Kaggle YT Face videos
https://www.kaggle.com/selfishgene/youtube-faces-with-facial-keypoints
Provides 3D key point annotations for the YT faces dataset, created with the [FAN of Bulat et al.](https://github.com/1adrianb/face-alignment). It required filtering of bad fits.

* WIDER FACE
http://shuoyang1213.me/WIDERFACE/index.html
For face detection. Provides bounding box annotations for a wide variety of images.

Usage
-----

There is no proper packaging yet. Therefore the project dir should be added to the python search path. The Anaconda environment is highly recommended. Assuming you have that, and work under Linux, and the dependencies are installed, you can get started by
```bash
cd dir/with/tracker-traincode
```
```bash
export PYTHONPATH=`pwd`  # Set python search path
export DATADIR=dir/with/data/files # our notebooks read this environment variable when looking for data sets
```

Regarding the datasets. Download them. Then run the following conversion scripts.
```bash
# Use -n <number> to limit the number of data points for faster development and testing
python scripts/dsaflw2k_processing.py    $@ $DATADIR/AFLW2000-3D.zip $DATADIR/aflw2k.h5
python scripts/ds300wlp_processing.py    $@ $DATADIR/300W-LP.zip $DATADIR/300wlp.h5
python scripts/dsytfaces_processing.py   $@ $DATADIR/YTFaces $DATADIR/ytfaces.h5
python scripts/dswiderface_processing.py $@ $DATADIR/wider_faces $DATADIR/widerfacessingle.h5
```

The script folder further contains notebooks for training and other tasks. Check the data with the help of  `DataVisualization.ipynb`. Run training in the notebooks `TrainLocalizer.ipynb` and `TrainKeypoints.ipynb`. The result can be inspected with `LocalizerEvaluation.ipynb` and `PoseNetworkEvaluation.ipynb`.

Afterwards, the networks must be converted to the ONNX format. ONNX is Microsofts storage format which is supported by a relatively lightweight runtime of the same name, allowing inference on the CPU. To carry out this conversion there is `export_model_onnx.py` in the scripts folder.

Dependencies
------------

```
Python, PyTorch, Jupyter, OpenCV, SciPy, H5py, Progressbar2, ONNX
```

Miscellaneous
------------

### Coordinate systems


In the world frame, X is forward from the point of view of the faces, pointing toward the camera. Y is up. Z is right. When viewed from the front, the face has identity rotation, meaning its local axes are aligned with the world axes.

The camera is considered fixed. In camera space, X is right, Y is down and Z is out of the screen. So, to get from world space to camera or image space, a respective transformation must be taken into account.

In future this should be simplified.

### OpenCV Performance

The OpenCV from Conda Forge run extremely slowly. Turns out it is better to force it to use only one thread. Hence the occasional lines
```
import cv2
cv2.setNumThreads(1)
```
The issue has been reported before. See there:
https://github.com/ContinuumIO/anaconda-issues/issues/10041
https://github.com/opencv/opencv/issues/11107#issuecomment-393475735


Licensing
---------

This software, I.e, everything not covered by other licenses is published under the ISC license.

Copyright 2021 Michael Welter

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

### Deformable Face Model

The modified BFM2009 face model in datasets/bfm_noneck_v3.pkl is only for academic use. For commercial use, you need to apply for the commercial license, some refs are below:

* https://faces.dmi.unibas.ch/bfm/?nav=1-0&id=basel_face_model
* https://faces.dmi.unibas.ch/bfm/bfm2019.html
* P. Paysan et al. (2009) "A 3D Face Model for Pose and Illumination Invariant Face Recognition"

### 3DDFA V2

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

### AFLW2000-3d / AFLW


AFLW2000-3d is derived from AFLW (https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/). The authors of AFLW2000-3d don't provide a license agreement. Hence, in the following the license of AFLW is reproduced:

License Agreement

By downloading the database you agree to the following restrictions:

* The AFLW database is available for non-commercial research purposes only.
* The AFLW database includes images obtained from FlickR which are not property of Graz University of Technology. Graz University of Technology is not responsible for the content nor the * meaning of these images. Any use of the images must be negociated with the respective picture owners, according to the Yahoo terms of use. In particular, you agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
* You agree not to further copy, publish or distribute any portion of the AFLW database. Except, for internal use at a single site within the same organization it is allowed to make copies of the database.
* All submitted papers or any publicly available text using the AFLW database must cite our paper
* The organization represented by you will be listed as users of the AFLW database.

### 300W-LP / 300W

300W-LP is derived from 300W (https://ibug.doc.ic.ac.uk/resources/300-W/). 300W explicitly prohibits commercial use as stated on the project website. Citation:
> The data are provided for research purposes only. Commercial use (i.e., use in training commercial algorithms) is not allowed.

### YouTube Faces With Facial Keypoints

Published under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) which is a permissive license. This data set is derived from [YouTube Faces](https://www.cs.tau.ac.il/~wolf/ytfaces/) for which I did not find terms of use.

### WIDER FACE

I did not find terms of use for this dataset specifically. However it is based on [WIDER](http://yjxiong.me/event_recog/WIDER/), where you have to agree to using the dataset for research purposes only before downloading.