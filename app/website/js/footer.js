fetch("../html/footer.html").then((res) => {
    return res.text();
}).then((data) => {
    const parser = new DOMParser();
    const page = parser.parseFromString(data, "text/html");
    const content = page.body.innerHTML;

    const container = document.querySelector(".footer-container");
    container.innerHTML = content;
})