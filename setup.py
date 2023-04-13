import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup,find_packages
from pathlib import Path
this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

if not torch.cuda.is_available():
    setup(
        name='deltamic',
        version='0.0.6',
        description='Differentiable rendering through a 3d fluorescence microscope',
        url='https://github.com/sacha-ichbiah/deltamic',
        author='Sacha Ichbiah',
        author_email='sacha.ichbiah@polytechnique.org',
        license='BSD',
        include_package_data=True,
        packages = find_packages(),

        install_requires=['numpy>=1.21.6',
                    'torch>=1.13.1',
                    'largesteps>=0.2.1',
                    'robust_laplacian>=0.2.2',
                    'scipy>=1.5.4',
                    'trimesh>=3.10.2',
                        ],
        long_description=readme,
        long_description_content_type="text/markdown"
    )

else : 

    setup(
        name='deltamic',
        version='0.0.6',
        description='Differentiable rendering through a 3d fluorescence microscope',
        url='https://github.com/sacha-ichbiah/deltamic',
        author='Sacha Ichbiah',
        author_email='sacha.ichbiah@polytechnique.org',
        license='BSD',
        include_package_data=True,
        packages = find_packages(),

        ext_modules=[
            CUDAExtension(
                'deltamic.fourier3d_cpp',
                sources = ['deltamic/cpp/fourier3d_cuda.cpp','deltamic/cpp/fourier3d_cuda_kernel.cu'],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']}),
        ],
        cmdclass={'build_ext': BuildExtension,
        },

        install_requires=['numpy>=1.21.6',
                    'torch>=1.13.1',
                    'largesteps>=0.2.1',
                    'robust_laplacian>=0.2.2',
                    'scipy>=1.5.4',
                    'trimesh>=3.10.2',
                        ],
        long_description=readme,
        long_description_content_type="text/markdown"
    )


"""
##CPP VERSION
if not torch.cuda.is_available():
    setup(
        name='deltamic',
        version='0.0.3',
        description='Differentiable rendering through a 3d fluorescence microscope',
        url='https://github.com/sacha-ichbiah/deltamic',
        author='Sacha Ichbiah',
        author_email='sacha.ichbiah@polytechnique.org',
        license='BSD',
        include_package_data=True,
        packages = find_packages(),

        ext_modules=[
            CppExtension(
                'deltamic.fourier3d_cpp',
                sources = ['deltamic/cpp/fourier3d.cpp'],
                extra_compile_args=['-g']),
        ],
        cmdclass={'build_ext': BuildExtension,
        },

        install_requires=['numpy>=1.21.6',
                    'torch>=1.13.1',
                    'largesteps>=0.2.1',
                    'robust_laplacian>=0.2.2',
                    'scipy>=1.5.4',
                    'trimesh>=3.10.2',
                        ],
        long_description=readme,
        long_description_content_type="text/markdown"
    )
"""