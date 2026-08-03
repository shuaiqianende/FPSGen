from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_3D',
    # ``pip install -e path/to/chamfer3D`` executes from this directory. Map
    # the current directory to the public ``chamfer3D`` package explicitly;
    # without package_dir, setuptools installs only a top-level extension and
    # ``from chamfer3D.dist_chamfer_3D import ...`` fails in a clean shell.
    packages=['chamfer3D'],
    package_dir={'chamfer3D': '.'},
    ext_modules=[
        CUDAExtension('chamfer3D.chamfer_3D', [
            'chamfer_cuda.cpp',
            'chamfer3D.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
