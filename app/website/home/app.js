document.addEventListener('DOMContentLoaded', function () {
  const startBtn = document.querySelector('.component .button-name');
  const formSection = document.querySelector('.form-section');
  const input = document.querySelector('.form-input');
  const analyzeBtn = document.querySelector('.button-name-wrapper .button-name-2');

function ensureErrorNode() {
  let err = formSection.querySelector('[data-error-msg]');
  if (!err) {
    err = document.createElement('div');
    err.setAttribute('data-error-msg', 'true');
    err.textContent = 'Isi dulu informasinya ya';
    err.style.display = 'none';
    err.style.color = '#d32f2f';
    err.style.marginTop = '6px';
    err.style.fontFamily = "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif";
    err.style.fontSize = '30px';
    err.style.fontWeight = '700';
    err.style.fontStyle = 'italic';
    formSection.appendChild(err);
  }
  return err;
}

  function scrollToInput() {
    if (formSection) {
      formSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setTimeout(() => input && input.focus({ preventScroll: true }), 250);
    }
  }

  if (startBtn) {
    startBtn.style.cursor = 'pointer';
    startBtn.addEventListener('click', scrollToInput);
  }

  if (analyzeBtn) {
    analyzeBtn.style.cursor = 'pointer';
    analyzeBtn.addEventListener('click', function () {
      const val = (input && input.value ? input.value : '').trim();
      const err = ensureErrorNode();
      if (!val) {
        err.style.display = 'block';
        input && input.focus();
        return;
      }
      err.style.display = 'none';
      window.location.href = '../home-check/index.html';
    });
  }

  if (input) {
    input.addEventListener('input', function () {
      const err = formSection && formSection.querySelector('[data-error-msg]');
      if (err && (input.value || '').trim()) err.style.display = 'none';
    });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') analyzeBtn && analyzeBtn.click();
    });
  }

  const navButtons = document.querySelectorAll('.navbar .classic-button-2');
  navButtons.forEach((el) => {
    const label = (el.textContent || '').trim().toLowerCase();
    el.style.cursor = 'pointer';
    el.setAttribute('tabindex', '0');
    el.setAttribute('role', 'button');
    let target = null;
    if (label === 'recent activity') target = '/activity/index.html';
    if (label === 'about us') target = '/about-us/index.html';
    if (!target) return;
    const go = () => (window.location.href = target);
    el.addEventListener('click', go);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        go();
      }
    });
  });
});
