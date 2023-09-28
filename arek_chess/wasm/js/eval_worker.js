importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.js");
importScripts("/onnxruntime/ort.js");

async function setupWorker() {
    console.log("setting up worker");

    self.pyodide = await loadPyodide();
    await self.pyodide.loadPackage("micropip");
    self.micropip = pyodide.pyimport("micropip");
    await self.micropip.install("../hackable_bot-0.0.4-py3-none-any.whl")

    self.board_pkg = pyodide.pyimport("arek_chess.board.hex.hex_board");
    self.pkg = pyodide.pyimport("arek_chess.workers.eval_worker");
    self.pkg.EvalWorker.callKwargs(null, null, null, null, null, 32, 1, self.board_pkg.HexBoard, 7, {evaluator_name: "hex"});

    // self.eval_worker_module = pyodide.pyimport("eval_worker");
    //   let results = await self.pyodide.runPythonAsync(`
    //     from arek_chess.board.chess.chess_board import ChessBoard
    //     from arek_chess.workers.eval_worker import EvalWorker
    //
    //     worker = EvalWorker()
    // `);

    console.log("... worker ready");
}

async function modelPredict() {
    ort.env.wasm.wasmPaths = 'http://localhost:8008/onnxruntime/';
    ort.env.wasm.proxy = true;
    const sessionOption = { executionProviders: ['wasm'] };

    // const sessionOption = { executionProviders: ['webgl'] };
    const session = await ort.InferenceSession.create('../my_ppo_model.onnx', sessionOption);
    input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    feeds = [];
    n = 10000
    for (i=0; i<n; i++)
        feeds.push({"input": new ort.Tensor("float32", input, [1, 9])});
    console.log('feeds built')
    console.time("test_timer");
    for (let i=0; i<n; i++) {
        await session.run(feeds[i]);
    }
    console.timeEnd("test_timer");
}

let evalWorkerPromise = setupWorker();

self.onmessage = async (event) => {
    // make sure loading is done
    await evalWorkerPromise;

    console.log('received in js worker: ');
    console.log(event.data);
    // The worker copies the context in its own "memory" (an object mapping name to values)
    // for (const key of Object.keys(context)) {
    //     self[key] = context[key];
    // }

    // try {
    //     self.postMessage({ id: context.a });
    // } catch (error) {
    //     console.log(error);
    //     self.postMessage({ error: error.message, id: 1000 });
    // }
};
