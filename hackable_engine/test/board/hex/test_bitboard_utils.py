# type: ignore
from pytest import mark

from hackable_engine.board.hex.bitboard_utils import generate_cells


@mark.parametrize(
    "bb, cells",
    [[3, [0, 1]], [65552, [4, 16]], [40564819207303340847894502572034, [1, 105]]],
)
def test_generate_cells(bb: int, cells: list[int]) -> None:
    assert list(generate_cells(bb)) == cells
