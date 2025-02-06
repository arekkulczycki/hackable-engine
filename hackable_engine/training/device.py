# -*- coding: utf-8 -*-
from enum import Enum


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    AUTO = "auto"
    XPU = "xpu"  # intel Arc GPU
