# src/ags/registry.py
class Registry:
    """A simple name→function (or class) registry."""
    def __init__(self, name: str):
        self._name = name
        self._fns = {}

    def register(self, key: str):
        """Decorator: @REGISTRY.register('name')"""
        def decorator(fn_or_cls):
            if key in self._fns:
                raise KeyError(f"{key} already registered in {self._name}")
            self._fns[key] = fn_or_cls
            return fn_or_cls
        return decorator

    def get(self, key: str):
        """Return registered fn_or_cls by name"""
        if key not in self._fns:
            raise KeyError(f"{key} not found in {self._name}. "
                           f"Available: {list(self._fns.keys())}")
        return self._fns[key]

    def list(self):
        """List all registered names"""
        return list(self._fns.keys())

    def __contains__(self, key):
        return key in self._fns

    def __repr__(self):
        return f"<Registry name={self._name} items={list(self._fns.keys())}>"

MODELS = Registry("MODELS")
DATASETS = Registry("DATASETS")
OPTIMIZERS = Registry("OPTIMIZERS")
SCHEDULERS = Registry("SCHEDULERS")
GRADS = Registry("GRADS")
LOSSES = Registry("LOSSES")