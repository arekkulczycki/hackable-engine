rm arek_chess/board/chess/*cpython*
rm arek_chess/board/hex/*cpython*
rm arek_chess/common/queue/items/*cpython*
rm arek_chess/criteria/evaluation/*.cpython*
rm arek_chess/criteria/evaluation/chess/*.cpython*
rm arek_chess/criteria/evaluation/hex/*.cpython*
rm arek_chess/criteria/selection/*.cpython*
rm arek_chess/game_tree/*.cpython*
rm arek_chess/training/envs/*.cpython*

#mypyc arek_chess/board/chess/chess_board.py
#mypyc arek_chess/board/hex/hex_board.py
#mypyc arek_chess/common/queue/items/base_item.py
#mypyc arek_chess/common/queue/items/control_item.py
#mypyc arek_chess/common/queue/items/distributor_item.py
#mypyc arek_chess/common/queue/items/eval_item.py
#mypyc arek_chess/common/queue/items/selector_item.py
#mypyc arek_chess/criteria/selection/fast_selector.py
#mypyc arek_chess/criteria/evaluation/base_eval.py
#mypyc arek_chess/criteria/evaluation/chess/square_control_eval.py
#mypyc arek_chess/criteria/evaluation/hex/simple_eval.py
#mypyc arek_chess/game_tree/node.py
#mypyc arek_chess/game_tree/traverser.py
#mypyc arek_chess/training/envs/square_control_env_util.py
#mypyc arek_chess/training/envs/square_control_env_single_action_util.py
