document.addEventListener("DOMContentLoaded", function () {
  var p1 = document.createElement("link");
  p1.rel = "preconnect";
  p1.href = "https://fonts.googleapis.com";
  document.head.appendChild(p1);

  var p2 = document.createElement("link");
  p2.rel = "preconnect";
  p2.href = "https://fonts.gstatic.com";
  p2.crossOrigin = "anonymous";
  document.head.appendChild(p2);

  var link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = "https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,400;0,700;1,400;1,700&display=swap";
  document.head.appendChild(link);

  var textarea = document.getElementById("info-input");
  var button = document.querySelector("button.component");
  var errorBox = document.getElementById("input-error");
  var errorText = errorBox ? (errorBox.querySelector("span") || errorBox) : null;

  function setErrorContainerLayout() {
    if (!errorBox) return;
    errorBox.style.display = "flex";
    errorBox.style.justifyContent = "flex-start";
    errorBox.style.alignItems = "center";
    errorBox.style.width = "100%";
    errorBox.style.padding = "10px 0 0 0";      // sejajar kiri kotak input (batas border)
    errorBox.style.marginTop = "8px";
    errorBox.style.textAlign = "left";
  }

  function applyErrorStyles() {
    if (!errorText) return;
    errorText.textContent = "Isi dulu informasinya ya";
    errorText.style.fontFamily = '"Poppins", sans-serif';
    errorText.style.fontWeight = "700";
    errorText.style.fontStyle = "italic";
    errorText.style.fontSize = "30px";
    errorText.style.color = "#FF0000";
    errorText.style.textAlign = "left";
  }

  function showError() {
    setErrorContainerLayout();
    applyErrorStyles();
    if (errorBox) errorBox.classList.add("show");
    if (textarea) {
      textarea.setAttribute("aria-invalid", "true");
      textarea.focus();
    }
  }

  function hideError() {
    if (errorBox) errorBox.classList.remove("show");
    if (textarea) textarea.removeAttribute("aria-invalid");
  }

  if (button) {
    button.addEventListener("click", function (e) {
      e.preventDefault();
      var v = (textarea && textarea.value ? textarea.value : "").trim();
      if (!v) showError(); else hideError();
    });
  }

  if (textarea) {
    textarea.addEventListener("input", function () {
      if (this.value.trim()) hideError();
    });
  }

  if (document.fonts && document.fonts.load) {
    document.fonts.load('italic 700 30px "Poppins"').then(applyErrorStyles);
  }
});
