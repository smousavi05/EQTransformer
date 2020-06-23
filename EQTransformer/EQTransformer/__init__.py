name='EQTransformer'

import importlib
import sys
import warnings

from EQTransformer.core.trainer import trainer 
from EQTransformer.core.tester import tester
from EQTransformer.core.predictor import predictor
from EQTransformer.core.mseed_predictor import mseed_predictor
from EQTransformer.core.EqT_utils import *
from EQTransformer.utils.associator import run_associator
from EQTransformer.utils.downloader import downloadMseeds, makeStationList, downloadSacs
from EQTransformer.utils.hdf5_maker import preprocessor
from EQTransformer.utils.plot import plot_detections, plot_data_chart

__all__ = ['core', 'utils', 'tests']
__version__ = '0.0.1'
_import_map = {}

class EqtDeprecationWarning(UserWarning):
    """
    Force pop-up of warnings.
    """
    pass

if sys.version_info.major < 3:
    raise NotImplementedError(
        "EqT no longer supports Python 2.x.")
    

class EqtRestructureAndLoad(object):
    """
    Path finder and module loader for transitioning
    """

    def find_module(self, fullname, path=None):
        # Compatibility with namespace paths.
        if hasattr(path, "_path"):
            path = path._path

        if not path or not path[0].startswith(__path__[0]):
            return None

        for key in _import_map.keys():
            if fullname.startswith(key):
                break
        else:
            return None
        return self

    def load_module(self, name):
        # Use cached modules.
        if name in sys.modules:
            return sys.modules[name]
        # Otherwise check if the name is part of the import map.
        elif name in _import_map:
            new_name = _import_map[name]
        else:
            new_name = name
            for old, new in _import_map.items():
                if not new_name.startswith(old):
                    continue
                new_name = new_name.replace(old, new)
                break
            else:
                return None

        # Don't load again if already loaded.
        if new_name in sys.modules:
            module = sys.modules[new_name]
        else:
            module = importlib.import_module(new_name)

        # Warn here as at this point the module has already been imported.
        warnings.warn("Module '%s' is deprecated and will stop working "
                      "with the next delphi version. Please import module "
                      "'%s' instead." % (name, new_name),
                      EqtDeprecationWarning)
        sys.modules[new_name] = module
        sys.modules[name] = module
        return module


sys.meta_path.append(EqtRestructureAndLoad())


