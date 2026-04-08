from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="virtual-try-on-ai",
    version="1.0.0",
    author="Your Name",
    description="Advanced AI Image Generator for Virtual Try-On with Pose and Clothing Changes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/virtual-try-on-ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.24.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pydantic>=2.0.0",
        "tqdm>=4.66.0",
    ],
)
