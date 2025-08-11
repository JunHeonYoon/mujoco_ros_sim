import importlib, sys, os, glob
from collections import defaultdict
from ament_index_python.packages import get_packages_with_prefixes
import importlib.abc
import importlib.machinery as _machinery 

_cpp_bindings = importlib.import_module(__name__ + ".bindings")
sys.modules["bindings"] = _cpp_bindings


PKG_FACTORIES: dict[str, set[str]] = defaultdict(set)

def _auto_scan():
    for pkg_name, prefix in get_packages_with_prefixes().items():
        plugdir = os.path.join(prefix, "lib", f"{pkg_name}_plugins")
        if not os.path.isdir(plugdir):
            continue
        for so in glob.glob(os.path.join(plugdir, "*.so")):
            _load_so_and_track(pkg_name, so)
    _refresh()


def _load_so_and_track(pkg_name: str, so_path: str) -> None:
    sys.modules.setdefault(f"{pkg_name}.bindings", _cpp_bindings)
    _cpp_bindings.load_plugin_library(so_path)
    _cpp_bindings.export_new_factories()

    for attr in dir(_cpp_bindings):
        if attr.endswith("_cpp"):
            attr = attr[:-4]
            PKG_FACTORIES[pkg_name].add(attr)

def _refresh() -> None:
    for pkg_name, names in PKG_FACTORIES.items():
        try:
            pkg_mod = importlib.import_module(pkg_name)
        except ImportError:
            continue

        for short in names:
            func_name = f"{short}_cpp"
            func_name = short
            func = getattr(_cpp_bindings, func_name, None)
            if func and not hasattr(pkg_mod, func_name):
                setattr(pkg_mod, func_name, func)

class _PluginPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in PKG_FACTORIES and fullname not in sys.modules:
            spec = _machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                return None
            spec.loader = _PluginPatchLoader(spec.loader)
            return spec
        return None

class _PluginPatchLoader(importlib.abc.Loader):
    def __init__(self, base_loader):
        self.base_loader = base_loader
    def create_module(self, spec):
        return self.base_loader.create_module(spec) if hasattr(self.base_loader, "create_module") else None
    def exec_module(self, module):
        self.base_loader.exec_module(module)
        for short in PKG_FACTORIES.get(module.__name__, set()):
            cpp_name = f"{short}_cpp"
            fn = getattr(_cpp_bindings, cpp_name, None)
            if fn and not hasattr(module, short):
                setattr(module, short, fn)


sys.meta_path.insert(0, _PluginPatchFinder())
_auto_scan()

from .controller_interface import ControllerInterface
from .mujoco_sim       import MujocoSimNode