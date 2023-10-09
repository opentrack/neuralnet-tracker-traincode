from setuptools import setup #, find_packages

setup(
    name='tracker-traincode',
    description='Generates large pose faces',
    author='Michael Welter',
    license='MIT Licence',
    packages=["trackertraincode"],
    zip_safe=False,
    install_requires=[
        # TODO: Anaconda compatibility and also a complete list?
        # 'h5py',
        # 'numpy',
        # 'scipy',
        # 'pytorch',
        # 'trimesh',
        # 'pyrender',
        # 'opencv',
        # 'kornia',
        # 'pillow',
        # 'torchvision',
        # 'pytorch-minimize',
        # 'pytorch3d',
        # 'facenet-pytorch',
    ],
    python_requires=">=3.9",
)