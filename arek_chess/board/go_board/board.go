package main

// #include <stdlib.h>
import "C"
import (
    "fmt"
    "time"
    "strconv"
    "strings"
    "unsafe"
    "github.com/notnil/chess"
)

//export get_board_data
func get_board_data(fenStrC *C.char, moveStrC *C.char) *C.char {
    fenStr := C.GoString(fenStrC)
    moveStr := C.GoString(moveStrC)

    fen, _ := chess.FEN(fenStr)
    game := chess.NewGame(fen, chess.UseNotation(chess.UCINotation{}))
    game.MoveStr(moveStr)

    pos := game.Position()

    moves := game.ValidMoves()

    resp := make([]string, len(moves) + 1)
    resp[0] = pos.String()
    for i, move := range moves {
        resp[i + 1] = move.String()
    }

    return C.CString(strings.Join(resp, ","))
}

//export get_moves
func get_moves(fenStrC *C.char) *C.char {
    fenStr := C.GoString(fenStrC)

    fen, _ := chess.FEN(fenStr)
    game := chess.NewGame(fen, chess.UseNotation(chess.UCINotation{}))

    moves := game.ValidMoves()

    resp := make([]string, len(moves))
    for i, move := range moves {
        resp[i] = move.String()
    }

    return C.CString(strings.Join(resp, ","))
}

//export get_mobility
func get_mobility(fenStrC *C.char, moveStrC *C.char) C.int {
    fenStr := C.GoString(fenStrC)
    moveStr := C.GoString(moveStrC)

    fen, _ := chess.FEN(fenStr)
    game := chess.NewGame(fen, chess.UseNotation(chess.UCINotation{}))
    game.MoveStr(moveStr)

    moves := game.ValidMoves()

    return C.int(len(moves))
}

//export get_mobility_and_fen
func get_mobility_and_fen(fenStrC *C.char, moveStrC *C.char) *C.char {
    fenStr := C.GoString(fenStrC)
    moveStr := C.GoString(moveStrC)

    fen, _ := chess.FEN(fenStr)
    game := chess.NewGame(fen, chess.UseNotation(chess.UCINotation{}))
    pos := game.Position()
    move, _ := chess.UCINotation{}.Decode(pos, moveStr)
    pos = pos.Update(move)

    fenStr1 := pos.String()
    fen1, _ := chess.FEN(fenStr1)
    game1 := chess.NewGame(fen1, chess.UseNotation(chess.UCINotation{}))
    moves1 := game1.ValidMoves()
    moves1_n := strconv.Itoa(len(moves1))

    slice := strings.Split(fenStr1, " ")
    slice[3] = "-"
    fenStr2 := strings.Join(slice, " ")
    if pos.Turn() == chess.White {
        fenStr2 = strings.Replace(fenStr2, " w ", " b ", 1)
    } else {
        fenStr2 = strings.Replace(fenStr2, " b ", " w ", 1)
    }
    fen2, _ := chess.FEN(fenStr2)

    game2 := chess.NewGame(fen2, chess.UseNotation(chess.UCINotation{}))
    moves2 := game2.ValidMoves()
    moves2_n := strconv.Itoa(len(moves2))

    if pos.Turn() == chess.White {
        return C.CString(fenStr1 + "," + moves1_n + "," + moves2_n)
    } else {
        return C.CString(fenStr1 + "," + moves2_n + "," + moves1_n)
    }
}

//export get_fen
func get_fen(fenStrC *C.char, moveStrC *C.char) *C.char {
    fenStr := C.GoString(fenStrC)
    moveStr := C.GoString(moveStrC)

    fen, _ := chess.FEN(fenStr)
    game := chess.NewGame(fen, chess.UseNotation(chess.UCINotation{}))
    pos := game.Position()
    move, _ := chess.UCINotation{}.Decode(pos, moveStr)

    pos = pos.Update(move)

    ret := C.CString(pos.String())
//     defer C.free(unsafe.Pointer(ret))
    return ret
}

//export gc
func gc(s *C.char) {
    C.free(unsafe.Pointer(s))
}

func main() {
    t0 := time.Now()
    for i := 0; i < 10000; i++ {
        fenStr := "rnbqkbnr/p1pp1ppp/8/1p2p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"
        slice := strings.Split(fenStr, " ")
        slice[3] = "-"
        fenStr2 := strings.Join(slice, " ")
        fen, _ := chess.FEN(fenStr2)
        game2 := chess.NewGame(fen, chess.UseNotation(chess.UCINotation{}))
        moves2 := game2.ValidMoves()

        resp := make([]string, len(moves2))
        for i, move := range moves2 {
            resp[i] = move.String()
        }
    }
    t1 := time.Now()
    fmt.Println(t1.Sub(t0))
//     fmt.Println(len(moves2))
//     fmt.Println(strings.Join(resp, ","))
}