async function ui_boot() {
    let bootBtn = document.getElementById("boot");
    bootBtn.disabled = true;

    let killBtn = document.getElementById("kill");
    killBtn.disabled = false;

    let sizeInput = document.getElementById("size");
    let value = parseInt(sizeInput.value)

    if (value > 0) {
        await boot(value);

        sizeInput.disabled = true;
        let setupBtn = document.getElementById("setup");
        let searchBtn = document.getElementById("search");
        setupBtn.disabled = false;
        searchBtn.disabled = false;
    } else {
        alert("must choose board size");
    }
}

async function ui_kill() {
    let bootBtn = document.getElementById("boot");
    let setupBtn = document.getElementById("setup");
    let searchBtn = document.getElementById("search");
    bootBtn.disabled = true;
    setupBtn.disabled = true;
    searchBtn.disabled = true;

    await kill();

    let killBtn = document.getElementById("kill");
    let sizeInput = document.getElementById("size")
    killBtn.disabled = true;
    bootBtn.disabled = false;
    sizeInput.disabled = false;
}

async function ui_setup() {
    let notation = document.getElementById("notation").value;
    let size = document.getElementById("size").value;
    console.log(notation, size);
//    await setup(notation, size)
}
