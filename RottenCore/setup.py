from setuptools import setup, find_packages

setup(
    name='RottenCore',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'scipy',
        'tqdm',
        'numba',
        'lpips @ git+https://github.com/richzhang/PerceptualSimilarity.git'
    ],
    entry_points={
        'console_scripts': [
            'rottencore=rottencore:main',
        ],
    },
    author='RottenCore Contributors',
    author_email='rottencore@example.com',
    description='A general-purpose video-to-glyph converter tool.',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RottenCore/RottenCore', # Placeholder for RottenCore's new repository
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License', # Changed to Apache 2.0
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Video',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)