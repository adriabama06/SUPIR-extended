from setuptools import setup, find_packages

setup(
    name="SUPiR",
    version="0.1.0",
    author="d8ahazard",
    author_email="donate.to.digitalhigh@gmail.com",
    description="A package containing SUPIR and sgm directories.",
    packages=find_packages(include=["SUPIR", "sgm", "SUPIR.*", "sgm.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch>=2.1.0,<3.0.0",
        "torchvision>=0.10.0",
        "numpy",
        "Pillow>=10.2.0",
        "tqdm",
        "pytorch_lightning",  # Leave this to be set by the installed torch version
        "einops",
        "requests",
        "facexlib>=0.3.0",
        "opencv-python",
        "huggingface_hub",
        "open-clip-torch>=2.24.0",
        "transformers>=4.38.2",
        "diffusers",
        "psutil",
        "matplotlib>=3.8.3",
        "omegaconf>=2.3.0",
        "packaging",
        "safetensors",
        "fsspec",
        "kornia"
    ],
    extras_require={
        'windows': [
            'triton @ https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl'
        ]
    },
    python_requires='>=3.9',
    options={
        'easy_install': {
            'index_url': 'https://download.pytorch.org/whl/cu121',
        }
    }
)
