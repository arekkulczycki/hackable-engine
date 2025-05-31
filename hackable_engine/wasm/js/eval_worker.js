importScripts("https://cdn.jsdelivr.net/pyodide/v0.28.0/full/pyodide.js");
importScripts("https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort.min.js");
//importScripts("/onnxruntime/ort.js");
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";


async function setupPyodide() {
    self.pyodide = await loadPyodide();
    await self.pyodide.loadPackage("numpy");
    await self.pyodide.loadPackage("micropip");
    self.micropip = pyodide.pyimport("micropip");
    await self.micropip.install("../hackable_engine-0.1.0-py3-none-any.whl")
}

async function setupJSWorker(boardSize) {
    self.ortSessionBlack = await ort.InferenceSession.create(
        `../${boardSize}black.onnx`,
        { executionProviders: ["wasm"] }
    );
    self.ortSessionWhite = await ort.InferenceSession.create(
        `../${boardSize}white.onnx`,
        { executionProviders: ["wasm"] }
    );
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

    let worker_locks_pkg = pyodide.pyimport("hackable_engine.workers.configs.worker_locks");
    let worker_locks = worker_locks_pkg.WorkerLocks()

    let worker_queues_pkg = pyodide.pyimport("hackable_engine.workers.configs.worker_queues");
    let worker_queues = worker_queues_pkg.WorkerQueues(self.distributor_queue, self.eval_queue, self.selector_queue, self.control_queue)

    let worker_config_pkg = pyodide.pyimport("hackable_engine.workers.configs.eval_worker_config");
    let worker_config = worker_config_pkg.EvalWorkerConfig(workerNum, board_pkg.HexBoard, boardInitKwargs.size)

    let worker_pkg = pyodide.pyimport("hackable_engine.workers.eval_worker");
    self.worker = worker_pkg.EvalWorker.callKwargs(
        worker_locks,
        worker_queues,
        {config: worker_config, memory: memoryArr});

    console.log("... eval worker ready");
}

async function modelPredict() {
    ort.env.wasm.wasmPaths = "http://localhost:8008/onnxruntime/";
    ort.env.wasm.proxy = true;
    const sessionOption = { executionProviders: ["wasm"] };

    // const sessionOption = { executionProviders: ["webgl"] };
    const session = await ort.InferenceSession.create("../my_ppo_model.onnx", sessionOption);
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
    if (event.data.type === "eval_queue") {
        self.eval_queue.inject_js(event.data.item);
    } else if (event.data.type === "eval_queue_bulk") {
        let items = event.data.items;
        items.forEach((item) => {
            self.eval_queue.inject_js(item);
        });
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
        await setupJSWorker(event.data.boardInitKwargs.size)
        await setupWorker(arr, event.data.worker_num, event.data.boardInitKwargs)
        self.postMessage({"type": "ready"})

        // self.worker._set_wasm_port(self.port);
        self.worker._set_selector_wasm_port(self.search_worker_port);

        await self.worker._run();
    }
}

self.onmessage = handle_event;

let pyodidePromise = setupPyodide();
