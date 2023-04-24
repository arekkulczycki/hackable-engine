rm arek_chess/board/board.cpython*
rm arek_chess/board/board__*
rm arek_chess/criteria/evaluation/base_eval.cpython*
rm arek_chess/criteria/evaluation/base_eval__*
rm arek_chess/criteria/evaluation/square_control_eval.cpython*
rm arek_chess/criteria/evaluation/square_control_eval__*
rm arek_chess/game_tree/node.cpython*
rm arek_chess/game_tree/node__*
rm arek_chess/game_tree/traverser.cpython*
rm arek_chess/game_tree/traverser__*

mypyc arek_chess/board/board.py
mypyc arek_chess/criteria/evaluation/base_eval.py
mypyc arek_chess/criteria/evaluation/square_control_eval.py
mypyc arek_chess/game_tree/node.py
mypyc arek_chess/game_tree/traverser.py
