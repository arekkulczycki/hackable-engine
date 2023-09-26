// importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.js");
// importScripts("https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.15.1/ort.min.js")
// importScripts(`${self.location.origin}/onnxruntime/ort.js`)

// async function loadPyodideAndPackages() {
//     self.pyodide = await loadPyodide();
//     await self.pyodide.loadPackage("micropip");
//     self.micropip = pyodide.pyimport("micropip");
//     await self.micropip.install("../dist/hackable_bot-0.0.2-py3-none-any.whl")
//
//     self.pkg = pyodide.pyimport("controller");
//     // self.pkg = pyodide.pyimport("onnx");
//
//     // console.log(ortWasmThreaded);
//     ort.env.wasm.wasmPaths = "../onnxruntime/"
//     ort.env.wasm.proxy = true;
//     const sessionOption = { executionProviders: ['wasm'], graphOptimizationLevel:'all' };
//     const session = await ort.InferenceSession.create('../my_ppo_model.onnx', sessionOption);
//     input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
//     feed = {"input": new ort.Tensor("float32", input, [1, 9])};
//     console.log('feed built')
//     console.time("test_timer");
//     output = await session.run(feed);
//     console.timeEnd("test_timer");
//     console.log(output);
// }
// let pyodideReadyPromise = loadPyodideAndPackages();

self.onmessage = async (event) => {
    // make sure loading is done
    // await pyodideReadyPromise;

    const { id, ...context } = event.data;
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
        self.postMessage({ results, id: context.a });
    } catch (error) {
        console.log(error);
        self.postMessage({ error: error.message, id: 1000 });
    }
};

console.log('worker ready')