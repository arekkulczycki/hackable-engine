rm arek_chess/board/*cpython*
rm arek_chess/common/queue/items/*cpython*
rm arek_chess/criteria/evaluation/*.cpython*
rm arek_chess/criteria/selection/*.cpython*
rm arek_chess/game_tree/*.cpython*

mypyc arek_chess/board/board.py
mypyc arek_chess/common/queue/items/control_item.py
mypyc arek_chess/common/queue/items/distributor_item.py
mypyc arek_chess/common/queue/items/eval_item.py
mypyc arek_chess/common/queue/items/selector_item.py
mypyc arek_chess/criteria/selection/fast_selector.py
mypyc arek_chess/criteria/evaluation/base_eval.py
mypyc arek_chess/criteria/evaluation/square_control_eval.py
mypyc arek_chess/game_tree/node.py
mypyc arek_chess/game_tree/traverser.py
