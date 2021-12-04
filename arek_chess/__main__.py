# -*- coding: utf-8 -*-
"""

"""

import time

from arek_chess.board.board import Board
from arek_chess.search_tree_manager import SearchTreeManager

if __name__ == "__main__":
    t0 = time.time()
    SearchTreeManager(Board(), True).run_search()
    print(time.time() - t0)
