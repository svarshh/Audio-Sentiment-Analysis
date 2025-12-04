document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("uploadForm");

    form.addEventListener("submit", async function(e) {
        e.preventDefault();

        const fileInput = document.getElementById("file");
        if (!fileInput.files.length) {
            alert("Please select a file to upload!");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const resultEl = document.getElementById("result");
        resultEl.textContent = "Analyzing...";

        try {
            const response = await fetch("/", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Network response was not ok");
            const text = await response.text();

            // Update the pre tag with the response HTML
            const parser = new DOMParser();
            const doc = parser.parseFromString(text, "text/html");
            const result = doc.getElementById("result").textContent;
            resultEl.textContent = result;

        } catch (err) {
            resultEl.textContent = "Error: " + err.message;
        }
    });
});
