# type: ignore
from hackable_engine.board.hex.move import Move


def test_move() -> None:
    assert Move(1, 13).mask == 1
    assert Move(1, 13).c == 0
    assert str(Move(1, 13)) == "a1"

    assert Move(4, 13) == Move.from_c(2, 13) == Move.from_xy(2, 0, 13)
    assert Move(17179869184, 13) == Move.from_c(34, 13) == Move.from_xy(8, 2, 13)
