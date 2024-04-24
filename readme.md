# OpNet: On the power of data augmentation for head pose estimation networks

A.K.A. OpenTrack's NeuralNet Tracker Training and Evaluation Code

Intro
-----

This branch contains the code for the publication. Beware, it also contains leftover things from past experiments.

This readme contains instructions for evaluation and training.

Install
-------

Setup a Python environment with a recent PyTorch. Tested with Python 3.11
and PyTorch 2.3.0. Using Python Anaconda:

```bash
# Create and activate python environment
conda create -p <path> python=3.11
conda activate <path>

# Install dependencies
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge numpy scipy opencv kornia matplotlib tqdm h5py onnx onnxruntime strenum tabulate

# Install `trackertraincode` from this repo in developer mode (as symlink)
cd <this repo>
pip install -e .
```

This should support training and eval. To generate the datasets you also need `pytorch-minimize facenet-pytorch scikit-learn trimesh pyrender`.

Set the `DATADIR` variable at least.
```bash
export DATADIR=<path to preprocessing outputs>
export NUM_WORKERS=<number of cpu cores> # For the data loader
```

Evaluation
----------

Download AFLW2000-3D from http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm.

Download a pytorch model checkpoint.

* Baseline Ensemble: https://drive.google.com/file/d/19LrssD36COWzKDp7akxtJFeVTcltNVlR/view?usp=sharing
* Additionally trained on Face Synthetics (BL+FS): https://drive.google.com/file/d/19zN8KICVEbLnGFGB5KkKuWrPjfet-jC8/view?usp=sharing
* Labeling Ensemble (RA-300W-LP from Table 3): https://drive.google.com/file/d/13LSi6J4zWSJnEzEXwZxr5UkWndFXdjcb/view?usp=sharing

### Option 1

Run `scripts/AFLW20003dEvaluation.ipynb`
It should give results pretty close to the paper. The face crop selection is different though and so the result won't be exactly the same.

### Option 2

Run the preprocessing and then the evaluation script.

```bash
# The output filename "aflw2k.h5" must batch the hardcoded value in "pipelines.py"
python scripts/dsaflw2k_processing.py $DATADIR/AFLW2000-3D.zip $DATADIR/aflw2k.h5`

# Will look in $DATADIR for aflw2k.h5.
python scripts/evaluate_pose_network.py --ds aflw2k3d <path to model(.onnx|.ckpt)>
```

It supports ONNX conversions as well as pytorch checkpoints. But the script must be adapted to the concrete model configuration for the checkpoint if that is used. If you wish to process the outputs further, like for averaging like in the paper, there is an option to generate json files.


Integration in OpenTrack
------------------------

https://github.com/opentrack/opentrack

 It currently has some older models though. Choose the "Neuralnet" tracker plugin.


Training
--------

Several datasets are used. All of which are preprocessed and the result (partially) stored in h5 files.

Rough guidelines for reproduction follow. First to get the data there is
the expositional script below which enumerates everything.

```bash
# 300W-LP
# Go to http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm and find the download for 300W-LP.zip.
# Currently it's on google drive with the ID as used below. Better check it yourself.
gdown 0B7OEHD3T4eCkVGs0TkhUWFN6N1k
# Note: gdown is a pip installable tool for downloading from google drive. You can ofc use anything you want.

# AFLW2000-3d
wget www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip

#LaPa Megaface 3D Labeled "Large Pose" Extension
#https://drive.google.com/file/d/1K4CQ8QqAVXj3Cd-yUt3HU9Z8o8gDmSEV/view?usp=drive_link
$ gdown 1K4CQ8QqAVXj3Cd-yUt3HU9Z8o8gDmSEV

#300W-LP Reproduction
#https://drive.google.com/file/d/1uEqba5JCGQMzrULnPHxf4EJa04z_yHWw/view?usp=drive_link
$ gdown 1uEqba5JCGQMzrULnPHxf4EJa04z_yHWw

#WFLW 3D Labeled "Large Pose" Extension
#https://drive.google.com/file/d/1SY33foUF8oZP8RUsFmcEIjq5xF5m3oJ1/view?usp=drive_link
$ gdown 1SY33foUF8oZP8RUsFmcEIjq5xF5m3oJ1

# Face Synthetics (https://github.com/microsoft/FaceSynthetics)
wget --tries=0 --continue --server-response --timeout=0 --retry-connrefused https://facesyntheticspubwedata.blob.core.windows.net/iccv-2021/dataset_100000.zip
```

Now some preprocessing and unpacking:

```bash
python scripts/dsprocess_aflw2k.py AFLW2000-3D.zip $DATADIR/aflw2k.h5

# Optional, for training on the original 300W-LP:
python scripts/dsprocess_300wlp.py --reconstruct-head-bbox 300W-LP.zip $DATADIR/300wlp.h5

# Face Synthetics 
python scripts/dsprocess_synface.py dataset_100000.zip $DATADIR/microsoft_synface_100000-v1.1.h5

# Custom datasets
unzip lapa-megaface-augmented-v2.zip -d ../$DATADIR/
unzip wflw_augmented_v4.zip -d ../$DATADIR/
unzip reproduction_300wlp-v12.zip -d ../$DATADIR/
```

The processed files can be inspected in the notebook `DataVisualization.ipynb`.

Now training should be possible. For the baseline it should be:
```bash
python scripts/train_poseestimator.py --lr 1.e-3 --epochs 1500 --ds "repro_300_wlp+lapa_megaface_lp:20000+wflw_lp" \
    --save-plot train.pdf \
    --with-swa \
    --with-nll-loss \
    --backbone mobilenetv1 \
    --no-onnx \
    --roi-override original \
    --no-blurpool \
    --outdir <output folder>
```

It will look at the environment variable `DATADIR` to find the datasets. Notable flags and settings are the following:

```bash
--with-nll-loss # Enables NLL losses
--backbone resnet18 # Use ResNet backbone
--no-blurpool # Disable use of Blur Pool instead of learnable strided conv.
--no-imgaug # Disable image intensity augmentation
--no-pointhead # Disable landmark predictions
--raug <value in degree> # Set the maximum angle of in-plane rotation augmentation. Zero disables it.


--ds "300wlp" # Train on original 300W-LP
--ds "300wlp:92+synface:8" # Train on Face Synthetics + 300W-LP
--ds "repro_300_wlp_woextra" # Train on the 300W-LP reproduction without the eye + illumination variations. (Needs unpublished dataset :-/)
--ds "repro_300_wlp" # Train only on the 300W-LP reproduction
--ds "repro_300_wlp+lapa_megaface_lp+wflw_lp+synface" # Train the "BL + FS" case which should give best performing models.
```

Creation of 3D Labeled WFLW & LaPa Large Pose Expansions
--------------------------------------------------------

* Preprocess LaPa and Megaface by scripts in `scripts/`.
* Download pseudo labeling ensemble.
* Generate pseudo labels
* Find the github repository face-3d-rotation-augmentation. Install the package in it with pip.
* Use the notebooks (in this repo) `scripts/DsLapaMegafaceFitFaceModel.ipynb`, `scripts/DsLapaMegafaceLargePoseCreation.ipynb`, `scripts/DsWflwFitFaceModel.ipynb` and `scripts/DsWflwLargePoseCreation.ipynb`.


Miscellaneous
-------------

### Coordinate System

It's a right handed system. Seen from the front, X is right, Y is down and Z is into the screen.
This coordinate system is used for world space and screen space. Also as local coordinate system
of the head, albeit the directions as described apply of course only at zero rotation.

### File format

Labels are stored in a HDF5 format. Input images maybe separated or integrated in the same file. Here is a dump of a
file with image included, where N is the number of samples:

```
/coords shape=(N, 3), dtype=float32
    ATTRIBUTE category: "xys" (str)
/images shape=(N,), dtype=object
    ATTRIBUTE category: "img" (str)
    ATTRIBUTE lossy: "True" (bool_)
    ATTRIBUTE storage: "varsize_image_buffer" (str)
/pt3d_68 shape=(N, 68, 3), dtype=float32
    ATTRIBUTE category: "pts" (str)
/quats shape=(N, 4), dtype=float32
    ATTRIBUTE category: "q" (str)
/rois shape=(N, 4), dtype=float32
    ATTRIBUTE category: "roi" (str)
/shapeparams shape=(N, 50), dtype=float16
    ATTRIBUTE category: "" (str)
```

As you can see the top level has several HDF5 Datasets (DS) with label data. `images` is the DS with the images.
The DS have attributes with metadata. There is the `category` which implies the kind of information stored in the DS.
The `image` DS has a `storage` attribute which tells if it the images stored inline or externally. `varsize_image_buffer`
means that the data type is a variabled sized buffer which holds the image. When `lossy` is true then the images are
encoded as JPG, else as PNG. When `storage` is set to `image_filename` then the DS contains relative paths to external
files. The other label fields are label data and should be relatively self-explanatory.

Relevant code for reading and writing those files can be found in `trackertraincode/datasets/dshdf5.py`, 
`trackertraincode/datasets/dshdf5pose.py` and the preprocessing scripts `scripts/dsprocess_*.py`.