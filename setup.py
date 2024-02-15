from setuptools import setup, find_namespace_packages


def _read(f):
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="gifr",
    description="gradio interfaces for Deep Learning Docker images that use Redis for receiving data to make predictions on.",
    long_description=(
            _read('DESCRIPTION.rst') + b'\n' +
            _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/waikato-datamining/gifr",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Utilities',
        'Programming Language :: Python :: 3',
    ],
    license='MIT License',
    package_dir={
        '': 'src'
    },
    packages=find_namespace_packages(where='src'),
    install_requires=[
        "gradio",
        "redis",
        "opex",
    ],
    version="0.0.3",
    author='Peter Reutemann',
    author_email='fracpete@waikato.ac.nz',
    entry_points={
        "console_scripts": [
            "gifr-imgcls=gifr.image_classification:sys_main",
            "gifr-imgseg=gifr.image_segmentation:sys_main",
            "gifr-objdet=gifr.object_detection:sys_main",
            "gifr-textclass=gifr.text_classification:sys_main",
            "gifr-textgen=gifr.text_generation:sys_main",
        ],
    },
)
