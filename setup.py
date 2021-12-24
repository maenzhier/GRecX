from setuptools import setup, find_packages

setup(
    name="grecx",
    python_requires='>3.5.0',
    version="0.0.3",
    author="Desheng Cai",
    author_email="caidsml@gmail.com",
    packages=find_packages(
        exclude=[
            'benchmarks',
            'data',
            'demo',
            'dist',
            'doc',
            'docs',
            'logs',
            'models',
            'test'
        ]
    ),
    install_requires=[
        "tf_geometric >= 0.0.77",
        "numpy >= 1.17.4",
        "tensorflow == 2.4.1",
        "scikit-learn >= 0.22",
        "tqdm",
        "Sphinx == 3.5.4",
        "faiss-cpu"
    ],
    extras_require={

    },
    package_data={'grecx': ["metrics/libranking.so", "metrics/libranking.dll"]},
    description="An Efficient and Unified Benchmark for GNN-based Recommendation.",
    license="GNU General Public License v3.0 (See LICENSE)",
    long_description=open("README.rst", "r", encoding="utf-8").read(),
    url="https://github.com/maenzhier/GRecX"
)