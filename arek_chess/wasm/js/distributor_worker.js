importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.js");

async function setupPyodide() {
    self.pyodide = await loadPyodide();
    await self.pyodide.loadPackage("micropip");
    self.micropip = pyodide.pyimport("micropip");
    await self.micropip.install("../hackable_bot-0.0.4-py3-none-any.whl")
}

async function setupWorker(memoryArray, boardInitKwargs) {
    let queue_pkg = pyodide.pyimport("arek_chess.common.queue.manager");
    let distributor_item_pkg = pyodide.pyimport("arek_chess.common.queue.items.distributor_item");
    let selector_item_pkg = pyodide.pyimport("arek_chess.common.queue.items.selector_item");
    let control_item_pkg = pyodide.pyimport("arek_chess.common.queue.items.control_item");
    let eval_item_pkg = pyodide.pyimport("arek_chess.common.queue.items.eval_item");

    let board_pkg = pyodide.pyimport("arek_chess.board.hex.hex_board");
    let board = board_pkg.HexBoard.callKwargs(boardInitKwargs);

    selector_item_pkg.SelectorItem.board_bytes_number = board.board_bytes_number
    distributor_item_pkg.DistributorItem.board_bytes_number = board.board_bytes_number
    eval_item_pkg.EvalItem.board_bytes_number = board.board_bytes_number

    self.selector_queue = queue_pkg.QueueManager(
        "selector_queue",
        selector_item_pkg.SelectorItem.loads,
        selector_item_pkg.SelectorItem.dumps
    );
    self.distributor_queue = queue_pkg.QueueManager(
        "distributor_queue",
        distributor_item_pkg.DistributorItem.loads,
        distributor_item_pkg.DistributorItem.dumps
    );
    self.control_queue = queue_pkg.QueueManager(
        "control_queue",
        control_item_pkg.ControlItem.loads,
        control_item_pkg.ControlItem.dumps
    );
    self.eval_queue = queue_pkg.QueueManager(
        "eval_queue",
        eval_item_pkg.EvalItem.loads,
        eval_item_pkg.EvalItem.dumps
    );

    let worker_pkg = pyodide.pyimport("arek_chess.workers.distributor_worker");
    self.worker = worker_pkg.DistributorWorker(
        null,
        null,
        null,
        self.distributor_queue,
        self.eval_queue,
        self.selector_queue,
        self.control_queue,
        board_pkg.HexBoard,
        boardInitKwargs.size,
        memoryArray  // TODO: not clear if this gets referenced or copied, make sure
    );

    console.log("... distributor worker ready");
}

self.eval_ports = [];
async function handle_event(event) {
    if (!event.data.type) {
        if (event.data.get("type") === "distributor_queue") {
            self.distributor_queue.inject_js(event.data.get("item"));
        } else if (event.data.get("type") === "distributor_queue_bulk") {
            let items = event.data.get("items");
            items.forEach((item) => {
                self.distributor_queue.inject_js(item);
            });
        } else {
            console.log("received else in distrib: ", event.data)
        }
    } else if (event.data.type === "distributor_port") {
        self.port = event.data.port;
        self.port.onmessage = async (eval_event) => {
            await handle_event(eval_event);
        }
    } else if (event.data.type === "control_port") {
        self.control_port = event.data.port;
    } else if (event.data.type === "eval_port") {
        self.eval_ports.push(event.data.port);
    } else if (event.data.type === "memory") {
        await pyodidePromise;

        var arr = new Int8Array(event.data.memory);
        await setupWorker(arr, event.data.boardInitKwargs)

        self.worker._set_eval_wasm_ports(self.eval_ports);
        self.worker._set_control_wasm_port(self.control_port);
        // self.worker._set_wasm_port(self.port);

        await self.worker._run();
    }
}

self.onmessage = handle_event

let pyodidePromise = setupPyodide();
