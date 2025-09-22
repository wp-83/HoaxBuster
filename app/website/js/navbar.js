fetch("../html/navbar.html").then((res) => {
    return res.text();
}).then((data) => {
    const parser = new DOMParser();
    const page = parser.parseFromString(data, "text/html");
    const content = page.body.innerHTML;
    
    const mainContent = document.querySelector(".nav-container");
    mainContent.innerHTML = content;

    updateClock();
    setInterval(updateClock, 1000);
    
    activePageStyle();
});

function updateClock() {
  const now = new Date();
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');
  document.getElementById("clock-now").textContent = `${hours}:${minutes}:${seconds} WIB`;
}

function activePageStyle(){
    const currentPath = window.location.href;
    const hyperlinks = document.querySelectorAll(".nav-container a");
    
    hyperlinks.forEach(hyperlink => {
        const path = new URL(hyperlink.href, window.location.origin).href;
        
        console.log(hyperlink);
        if (currentPath === path) {
            hyperlink.classList.add("active-page-button");
        }
    });
}