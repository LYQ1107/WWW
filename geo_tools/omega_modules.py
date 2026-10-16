"""Load geometry utilities without importing GMT/detectron2 registrations."""
import importlib
import sys
import types
from omega_common import ROOT

NAME = 'omega_standalone_geometry'
if NAME not in sys.modules:
    package = types.ModuleType(NAME)
    package.__path__ = [str(ROOT / 'gtr/geometry')]
    sys.modules[NAME] = package


def geometry_module(name):
    return importlib.import_module(NAME + '.' + name)
