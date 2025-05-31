async function setupWorkers(numEvalWorkers, boardSize) {
    const boardInitKwargs = {notation: "", size: boardSize};

    const sToDChannel = new MessageChannel();
    const dToSChannel = new MessageChannel();

    self.searchWorker = new Worker('./js/search_worker.js');
    self.searchWorker.onmessage = handleSearchMessage;
    self.distributorWorker = new Worker('./js/distributor_worker.js');

    self.searchWorker.postMessage({"type": "distributor_port", "port": sToDChannel.port1}, [sToDChannel.port1])
    self.distributorWorker.postMessage({"type": "distributor_port", "port": sToDChannel.port2}, [sToDChannel.port2])

    self.searchWorker.postMessage({"type": "control_port", "port": dToSChannel.port1}, [dToSChannel.port1])
    self.distributorWorker.postMessage({"type": "control_port", "port": dToSChannel.port2}, [dToSChannel.port2])

    self.evalWorkers = [];
    self.eToSChannels = [];
    self.dToEChannels = [];
    for (let i=0;i<numEvalWorkers;i++) {
        let worker = new Worker('./js/eval_worker.js')
        worker.onmessage = handleEvalMessage;

        let fromChannel = new MessageChannel();
        let toChannel = new MessageChannel();
        self.evalWorkers.push(worker);
        self.dToEChannels.push(toChannel);
        self.eToSChannels.push(fromChannel);

        worker.postMessage({"type": "search_port", "port": fromChannel.port1}, [fromChannel.port1])
        self.searchWorker.postMessage({"type": "search_port", "port": fromChannel.port2}, [fromChannel.port2])

        self.distributorWorker.postMessage({"type": "eval_port", "port": toChannel.port1}, [toChannel.port1])
        worker.postMessage({"type": "eval_port", "port": toChannel.port2}, [toChannel.port2])
    }

    var buff = new SharedArrayBuffer(1024);
    self.searchWorker.postMessage({"type": "memory", "memory": buff, "boardInitKwargs": boardInitKwargs})
    self.distributorWorker.postMessage({"type": "memory", "memory": buff, "boardInitKwargs": boardInitKwargs})

    for (let i=0;i<numEvalWorkers;i++) {
        self.evalWorkers[i].postMessage({"type": "memory", "memory": buff, "worker_num": i+1, "boardInitKwargs": boardInitKwargs})
    }
}

const numEvalWorkers = 4;
var workersReady = 0;
async function loadHackableBot(boardSize) {
    await setupWorkers(numEvalWorkers, boardSize);
    await waitUntil(allWorkersReady);

    console.log("all workers ready!");
}

function allWorkersReady() {
    return workersReady === numEvalWorkers;
}

async function waitUntil(condition) {
    await new Promise(resolve => {
        const interval = setInterval(() => {
            if (condition()) {
                clearInterval(interval);
                resolve();
            }
        }, 1000);
    });
}

async function handleSearchMessage(event) {
    console.log(event);
}

async function handleEvalMessage(event) {
    if (event.data.type === "ready") {
        workersReady++;
    }
}

async function setup(notation, size) {
    self.searchWorker.postMessage({"type": "setup", "boardInitKwargs": {notation: notation, size: size}});
}

async function search() {
    self.searchWorker.postMessage({"type": "search"});
}

async function boot(boardSize) {
    await loadHackableBot(boardSize);
}

async function kill() {
    self.evalWorkers.forEach((worker) => {
        worker.terminate();
    });
    self.distributorWorker.terminate()
    self.searchWorker.terminate();
}
