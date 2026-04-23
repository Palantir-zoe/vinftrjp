from .change_point import ChangePoint
from .fa import FA
from .sas import SAS
from .vs import VS
from .vsc import VSC

PROBLEM = {"sas": SAS, "fa": FA, "vs": VS, "vsc": VSC, "changepoint": ChangePoint, "change_point": ChangePoint}


def get_problem(name: str, *args, **kwargs):
    name = name.lower()

    if name.lower() not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name.lower()](*args, **kwargs)
