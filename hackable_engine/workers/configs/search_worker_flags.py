# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass(slots=True)
class SearchWorkerFlags:
    started: bool = False
    finished: bool = False
    should_profile: bool = False
    debug: bool = False
