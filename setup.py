""" Setup file """
from setuptools import setup
import versioneer

setup(
    name = "veco-opeco-modelo-opcion-pagos",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
