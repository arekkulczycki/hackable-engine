importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.js");
importScripts("/onnxruntime/ort.js");

async function setupPyodide() {
    self.pyodide = await loadPyodide();
    await self.pyodide.loadPackage("micropip");
    self.micropip = pyodide.pyimport("micropip");
    await self.micropip.install("../hackable_bot-0.0.4-py3-none-any.whl")
}

async function setupWorker(memoryArr, workerNum, boardInitKwargs) {
    let queue_pkg = pyodide.pyimport("hackable_engine.common.queue.manager");
    let eval_item_pkg = pyodide.pyimport("hackable_engine.common.queue.items.eval_item");
    let selector_item_pkg = pyodide.pyimport("hackable_engine.common.queue.items.selector_item");
    let distributor_item_pkg = pyodide.pyimport("hackable_engine.common.queue.items.distributor_item");

    let board_pkg = pyodide.pyimport("hackable_engine.board.hex.hex_board");
    let board = board_pkg.HexBoard.callKwargs(boardInitKwargs);

    eval_item_pkg.EvalItem.board_bytes_number = board.board_bytes_number
    selector_item_pkg.SelectorItem.board_bytes_number = board.board_bytes_number
    distributor_item_pkg.DistributorItem.board_bytes_number = board.board_bytes_number

    self.eval_queue = queue_pkg.QueueManager(
        "eval_queue",
        eval_item_pkg.EvalItem.loads,
        eval_item_pkg.EvalItem.dumps
    );
    self.selector_queue = queue_pkg.QueueManager(
        "selector_queue",
        selector_item_pkg.SelectorItem.loads,
        selector_item_pkg.SelectorItem.dumps
    );

    let worker_pkg = pyodide.pyimport("hackable_engine.workers.eval_worker");
    self.worker = worker_pkg.EvalWorker.callKwargs(
        null,
        null,
        null,
        self.eval_queue,
        self.selector_queue,
        workerNum,
        board_pkg.HexBoard,
        boardInitKwargs.size,
        {evaluator_name: "hex", memory: memoryArr});

    console.log("... eval worker ready");
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
    for (let i=0; i<n; i++)
        feeds.push({"input": new ort.Tensor("float32", input, [1, 9])});
    console.log('feeds built')
    console.time("test_timer");
    for (let i=0; i<n; i++) {
        await session.run(feeds[i]);
    }
    console.timeEnd("test_timer");
}

async function handle_event(event) {
    if (!event.data.type) {
        if (event.data.get("type") === "eval_queue") {
            self.eval_queue.inject_js(event.data.get("item"));
        } else if (event.data.get("type") === "eval_queue_bulk") {
            let items = event.data.get("items");
            items.forEach((item) => {
                self.eval_queue.inject_js(item);
            });
        } else {
            console.log("received else in eval: ", event.data)
        }
    } else if (event.data.type === "search_port") {
        self.search_worker_port = event.data.port;
    } else if (event.data.type === "eval_port") {
        self.port = event.data.port;
        self.port.onmessage = async (eval_event) => {
            await handle_event(eval_event);
        }
    } else if (event.data.type === "memory") {
        await pyodidePromise;

        var arr = new Int8Array(event.data.memory);
        await setupWorker(arr, event.data.worker_num, event.data.boardInitKwargs)
        self.postMessage({"type": "ready"})

        // self.worker._set_wasm_port(self.port);
        self.worker._set_selector_wasm_port(self.search_worker_port);

        await self.worker._run();
    }
}

self.onmessage = handle_event;

let pyodidePromise = setupPyodide();
