console.log("Dog Breed Identification Website Loaded");
function previewImage(event) {
    var reader = new FileReader();
    reader.onload = function() {
        var output = document.getElementById('preview');
        output.src = reader.result;
        output.style.display = "block";
    };
    reader.readAsDataURL(event.target.files[0]);
}


// Only alert for Inspect button
document.querySelectorAll(".inspect-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        alert("Inspection Feature Coming Soon!");
    });
});
