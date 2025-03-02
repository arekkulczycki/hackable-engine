# -*- coding: utf-8 -*-
from enum import IntEnum


class EdgeType(IntEnum):
    VERTICAL = 0
    """Along white edge."""
    HORIZONTAL = 1
    """Along black edge."""
    DIAGONAL = 2
    """Along short diagonal."""
