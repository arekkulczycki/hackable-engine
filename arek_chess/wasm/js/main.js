async function startWorker() {
  const pyodideWorker = new Worker('./js/nocache.js');

  const callbacks = {};

  pyodideWorker.onmessage = (event) => {
    const {id, ...data} = event.data;
    console.log(data);
    const onSuccess = callbacks[id];
    delete callbacks[id];
    onSuccess(data);
  };
}

async function loadHackableBot() {
  self.pyodide = await loadPyodide();
  await self.pyodide.loadPackage("micropip");
  self.micropip = pyodide.pyimport("micropip");
  await self.micropip.install("./hackable_bot-0.0.3-py3-none-any.whl")

  // chess_board_module = pyodide.pyimport("arek_chess.board.chess.chess_board");
  // sqc_eval_module = pyodide.pyimport("arek_chess.criteria.evaluation.chess.square_control_eval");
  //   let board = chess_board_module.ChessBoard();
  //   let sqc_eval = sqc_eval_module.SquareControlEval();
  //   console.log(sqc_eval.get_score(board, false));
  let results = await self.pyodide.runPythonAsync(`
    from arek_chess.board.chess.chess_board import ChessBoard
    from arek_chess.criteria.evaluation.chess.square_control_eval import SquareControlEval
    
    board = ChessBoard()
    evaluator = SquareControlEval()
    float(evaluator.get_score(board, False))
  `);
  console.log(results)
  pyodideWorker.postMessage({results: results, id: 0});
}

// const asyncRun = (() => {
//   let id = 0; // identify a Promise
//
//   return (script, context) => {
//     // the id could be generated more carefully
//     id = (id + 1) % Number.MAX_SAFE_INTEGER;
//
//     return new Promise((onSuccess) => {
//       callbacks[id] = onSuccess;
//       pyodideWorker.postMessage({
//         ...context,
//         id,
//       });
//     });
//   };
// })();

// export { asyncRun };
