# SYSTEM CREATORS ===
_SYSTEM_CREATORS = {}


def register_system_creator_cls(name):
    global _SYSTEM_CREATORS

    def _wrapper(creator):
        _SYSTEM_CREATORS[name] = creator
        return creator

    return _wrapper


def get_system_creator_cls(name):
    if not _SYSTEM_CREATORS.get(name):
        raise ValueError(
            f"{name} is not a registered system creator\n"
            f"Registered system creators are {list(_SYSTEM_CREATORS.keys())}"
        )
    return _SYSTEM_CREATORS[name]


# EVALUATOR CREATORS ===
_EVALUATOR_CREATORS = {}


def register_evaluator_creator(name):
    global _EVALUATOR_CREATORS

    def _wrapper(creator):
        _EVALUATOR_CREATORS[name] = creator
        return creator

    return _wrapper


def get_evaluator_creator(name):
    if not _EVALUATOR_CREATORS.get(name):
        raise ValueError(
            f"{name} is not a registered evaluator creator\n"
            f"Registered evaluator creators are {list(_EVALUATOR_CREATORS.keys())}"
        )
    return _EVALUATOR_CREATORS[name]


# NETWORKS RESTORERS ===
_NETWORKS_RESTORERS = {}


def register_networks_restorer(name):
    global _NETWORKS_RESTORERS

    def _wrapper(restorer):
        _NETWORKS_RESTORERS[name] = restorer
        return restorer

    return _wrapper


def get_networks_restorer(name):
    if not _NETWORKS_RESTORERS.get(name):
        raise ValueError(
            f"{name} is not a registered networks restorer\n"
            f"Registered networks restorer are {list(_NETWORKS_RESTORERS.keys())}"
        )
    return _NETWORKS_RESTORERS[name]


# FOCAL NETWORKS SETTERS ===
_FOCAL_NETWORKS_SETTERS = {}


def register_focal_networks_setter(name):
    global _FOCAL_NETWORKS_SETTERS

    def _wrapper(setter):
        _FOCAL_NETWORKS_SETTERS[name] = setter
        return setter

    return _wrapper


def get_focal_networks_setter(name):
    if not _FOCAL_NETWORKS_SETTERS.get(name):
        raise ValueError(
            f"{name} is not a registered focal networks setter\n"
            f"Registered focal networks setters are {list(_FOCAL_NETWORKS_SETTERS.keys())}"
        )
    return _FOCAL_NETWORKS_SETTERS[name]
