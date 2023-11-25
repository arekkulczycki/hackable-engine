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

    let worker_pkg = pyodide.pyimport("arek_chess.workers.search_worker");
    self.worker = worker_pkg.SearchWorker(null, null, null, self.selector_queue, self.distributor_queue, self.control_queue, memoryArray);
    self.worker.reset(board)

    console.log("... search worker ready");
}

self.ports = [];
async function handle_event(event) {
    if (!event.data.type) {
        if (event.data.get("type") === "selector_queue") {
            self.selector_queue.inject_js(event.data.get("item"));
        } else if (event.data.get("type") === "control_queue") {
            self.control_queue.inject_js(event.data.get("item"));
        } else if (event.data.get("type") === "selector_queue_bulk") {
            let items = event.data.get("items");
            items.forEach((item) => {
                self.selector_queue.inject_js(item);
            });
            // for (let i=0;i<items.count;i++) {
            //     self.selector_queue.inject_js(items[i]);
            // }
        } else {
            console.log("received else in search: ", event.data)
        }
    } else if (event.data.type === "distributor_port") {
        self.distributor_worker_port = event.data.port;
    } else if (event.data.type === "control_port" || event.data.type === "search_port") {
        self.ports.push(event.data.port);
        event.data.port.onmessage = async (eval_event) => {
            await handle_event(eval_event);
        }
    } else if (event.data.type === "memory") {
        // make sure loading is done
        await pyodidePromise;

        var arr = new Int8Array(event.data.memory);
        await setupWorker(arr, event.data.boardInitKwargs)

        // self.worker._set_wasm_ports(self.ports);
        self.worker._set_distributor_wasm_port(self.distributor_worker_port);
    } else if (event.data.type === "reset") {
        let board_pkg = pyodide.pyimport("arek_chess.board.hex.hex_board");
        let board = board_pkg.HexBoard.callKwargs(event.data.boardInitKwargs);
        await self.worker.reset(board);
    } else if (event.data.type === "search") {
        self.postMessage({"type": "move", "move": await self.worker.search()});
    }
}

self.onmessage = handle_event

let pyodidePromise = setupPyodide();
