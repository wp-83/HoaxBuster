document.addEventListener('DOMContentLoaded', () => {
  const startBtn = document.querySelector("#next-section");
  const formSection = document.querySelector('.second-section');
  const input = document.querySelector('#input-info');
  const analyzeBtn = document.querySelector('#check-info');
  const errorMsg = document.querySelector('.error-message');

  function scrollToInput() {
    if (!formSection) return;
    formSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setTimeout(() => input?.focus({ preventScroll: true }), 250);
  }

  async function runAnalysis() {
    const textToAnalyze = (input?.value || '').trim();

    if (!textToAnalyze) {
      errorMsg.textContent = 'Isi dulu informasinya ya';
      errorMsg.style.display = 'block';
      input?.focus();
      return;
    }
    errorMsg.style.display = 'none';

    try {
      const response = await fetch("https://william83.pythonanywhere.com/predict", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ information: textToAnalyze }),
      });

      if (!response.ok) throw new Error("Server error: " + response.status);

      const result = await response.json();
      const prediction = parseFloat(result.prediction);

      const analysisResult = {
        originalText: textToAnalyze,
        hoaxPercentage: isNaN(prediction) ? 0 : prediction * 100,
        timestamp: new Date().toISOString()
      };

      localStorage.setItem('analysisResult', JSON.stringify(analysisResult));

      let history = [];
      try {
        history = JSON.parse(localStorage.getItem('searchHistory')) || [];
        if (!Array.isArray(history)) history = [];
      } catch {
        history = [];
      }
      history.unshift(analysisResult);
      history = history.slice(0, 10);
      localStorage.setItem('searchHistory', JSON.stringify(history));

      window.location.href = 'result.html';

    } catch (error) {
      console.error('Error saat memanggil API:', error);
      alert('Terjadi kesalahan saat menganalisis. Silakan coba lagi.');
    }
  }

  if (startBtn) {
    startBtn.style.cursor = 'pointer';
    startBtn.addEventListener('click', scrollToInput);
  }

  if (analyzeBtn) {
    analyzeBtn.style.cursor = 'pointer';
    analyzeBtn.addEventListener('click', runAnalysis);
  }


  if (input) {
    input.addEventListener('input', () => {
      if (errorMsg && input.value.trim()) {
        errorMsg.style.display = 'none';
      }
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        runAnalysis(); 
      }
    });
  }
});
