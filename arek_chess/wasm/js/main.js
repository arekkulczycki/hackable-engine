async function setupWorkers(numEvalWorkers) {
  const boardInitKwargs = {notation: "", size: 13};

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

async function setupPyodide() {
  console.log("setting up pyodide...");

  self.pyodide = await loadPyodide();
  await self.pyodide.loadPackage("micropip");
  self.micropip = pyodide.pyimport("micropip");
  await self.micropip.install("../hackable_bot-0.0.4-py3-none-any.whl")
}

const numEvalWorkers = 4;
var workersReady = 0;
async function loadHackableBot() {
  await setupWorkers(numEvalWorkers);
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

async function reset() {  // TODO: pass in the notation and size
  self.searchWorker.postMessage({"type": "reset", "boardInitKwargs": {notation: "", size: 13}});
}

async function search() {
  self.searchWorker.postMessage({"type": "search"});
}

async function handleSearchMessage(event) {
  console.log(event.data);
}

async function handleEvalMessage(event) {
  if (event.data.type === "ready") {
    workersReady++;
  }
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

// let pyodidePromise = setupPyodide();
loadHackableBot();
