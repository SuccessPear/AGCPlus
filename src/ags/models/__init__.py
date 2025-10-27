from ags.registry import MODELS  # already defined in your repo

def build_model(cfg):
    name = cfg.model.name.lower()
    try:
        builder = MODELS.get(name)
    except KeyError:
        raise ValueError(f"Unknown model: {name}. Registered: {list(MODELS._fns.keys())}")
    return builder(cfg)
