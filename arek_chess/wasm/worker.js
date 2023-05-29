importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.js");

async function loadPyodideAndPackages() {
    self.pyodide = await loadPyodide();
    await self.pyodide.loadPackage("micropip");
    self.micropip = pyodide.pyimport("micropip");
    await self.micropip.install("./dist/hackable_bot-0.0.1-py3-none-any.whl")

    // await pyodide.runPythonAsync(`
    //     from pyodide.http import pyfetch
    //     response = await pyfetch("https://.../script.py")
    //     with open("script.py", "wb") as f:
    //         f.write(await response.bytes())
    // `)
    self.pkg = pyodide.pyimport("controller");
}
let pyodideReadyPromise = loadPyodideAndPackages();

self.onmessage = async (event) => {
    // make sure loading is done
    await pyodideReadyPromise;

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
        let ctrlr = self.pkg.Controller();
        ctrlr.boot_up();
        let results = ctrlr.get_response()
        self.postMessage({ results, id: context.a });
    } catch (error) {
        console.log(error);
        self.postMessage({ error: error.message, id: 1000 });
    }
};

console.log('worker ready')