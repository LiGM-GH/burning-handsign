/**
 * @param {Function} fun 
 */
function call_closure(fun) {
    fun()
}

function send_model() {
    let input = document.getElementById("fileInput");
    if (!(input instanceof HTMLInputElement)) {
        return;
    }

    let files = input.files;

    if (files == null || files == undefined) {
        return;
    }

    call_closure(async () => {
        const response = await fetch("/todos/" + todo_id, {
            method: "POST",
            body: params,
        });

        if (response == null || response == undefined || !(response instanceof Response)) {
            return;
        }

        console.log(response.text)
    })
}
