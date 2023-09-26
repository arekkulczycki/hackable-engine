importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.js");
importScripts("/onnxruntime/ort.js");

async function loadPyodideAndPackages() {
    self.pyodide = await loadPyodide();
    await self.pyodide.loadPackage("micropip");
    self.micropip = pyodide.pyimport("micropip");
    await self.micropip.install("../hackable_bot-0.0.3-py3-none-any.whl")

    // self.eval_worker_module = pyodide.pyimport("eval_worker");
      let results = await self.pyodide.runPythonAsync(`
        from arek_chess.board.chess.chess_board import ChessBoard
        from arek_chess.workers.eval_worker import EvalWorker

        worker = EvalWorker()
    `);

    console.log("pyodide worker");
    await modelPredict();
    console.log("model predicted in worker");
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

let pyodideReadyPromise = loadPyodideAndPackages();

self.onmessage = async (event) => {
    // make sure loading is done
    await pyodideReadyPromise;

    const { id, ...context } = event.data;
    console.log(id, context);
    // The worker copies the context in its own "memory" (an object mapping name to values)
    // for (const key of Object.keys(context)) {
    //     self[key] = context[key];
    // }


    try {
        // let results = await self.pyodide.runPythonAsync(`
        //     from controller import Controller
        //     controller = Controller()
        //     controller.boot_up()
        //     0
        // `);
        // let ctrlr = self.pkg.Controller();
        // ctrlr.boot_up();
        // let results = ctrlr.get_response()
        self.postMessage({ id: context.a });
    } catch (error) {
        console.log(error);
        self.postMessage({ error: error.message, id: 1000 });
    }
};
