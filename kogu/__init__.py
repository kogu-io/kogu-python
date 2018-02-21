from .kogu import Kogu, KoguException
from .jupytermagic import KoguExecution


def load_ipython_extension(ipython):
    ipython.register_magics(KoguExecution)
