document.addEventListener('DOMContentLoaded', function() {
    const resultTitle = document.getElementById('result-title');
    const resultDescription = document.getElementById('result-description');
    const analyzeBtn = document.querySelector('.analyze-btn');

    // Fungsi easing (bounce ringan)
    function easeOutBack(t) {
        const c1 = 1.5;
        const c3 = c1 + 1;
        return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    }

    // === Gauge Needle Overlay ===
    function createNeedleOverlay(chart, targetValue, duration = 2500) {
        const canvas = chart.canvas;
        const parent = canvas.parentNode;

        // Buat canvas overlay untuk jarum
        const overlay = document.createElement("canvas");
        overlay.width = canvas.width;
        overlay.height = canvas.height;
        overlay.style.position = "absolute";
        overlay.style.left = canvas.offsetLeft + "px";
        overlay.style.top = canvas.offsetTop + "px";
        parent.appendChild(overlay);

        const ctx = overlay.getContext("2d");

        // Animasi jarum
        let startTime = null;
        const fromValue = 0;
        const toValue = targetValue;

        function drawNeedle(value) {
            ctx.clearRect(0, 0, overlay.width, overlay.height);

            const { chartArea: { left, width, height } } = chart;

            const dataMin = 0;
            const dataMax = 100;
            const totalAngle = 270;
            const startAngle = -135;

            const normalizedValue = (value - dataMin) / (dataMax - dataMin);
            const angle = (normalizedValue * totalAngle + startAngle) * Math.PI / 180;

            ctx.save();
            ctx.translate(left + width / 2, 400);
            ctx.rotate(angle);

            // Jarum
            ctx.beginPath();
            ctx.fillStyle = "black";
            ctx.moveTo(-32, 0);
            ctx.lineTo(32, 0);
            ctx.lineTo(0, -height * 0.43);
            ctx.closePath();
            ctx.fill();

            // Lingkaran pangkal
            ctx.beginPath();
            ctx.arc(0, 0, 40, 0, 2 * Math.PI);
            ctx.fillStyle = "#6E433D";
            ctx.fill();

            ctx.restore();
        }

        function animate(time) {
            if (!startTime) startTime = time;
            const elapsed = time - startTime;
            const t = Math.min(elapsed / duration, 1);
            const eased = easeOutBack(t);

            const currentValue = fromValue + (toValue - fromValue) * eased;
            drawNeedle(currentValue);

            if (t < 1) requestAnimationFrame(animate);
        }

        requestAnimationFrame(animate);
    }

    // === Gauge Emoji Plugin ===

    const gaugeEmoji = {
        id: 'gaugeEmoji',
        afterDatasetDraw(chart, args, options) {
            const { ctx, chartArea: { width, height } } = chart;

            const emojis = [
                { icon: 'f580', value: 24 },
                { icon: 'f118', value: 52 },
                { icon: 'f11a', value: 84 },
                { icon: 'f119', value: 114 },
                { icon: 'f5c2', value: 143 },
            ];

            const dataMin = 0;
            const dataMax = 100;
            const outerRadius = chart.getDatasetMeta(0).data[0].outerRadius;
            const innerRadius = chart.getDatasetMeta(0).data[0].innerRadius;
            const emojiRadius = (outerRadius + innerRadius) / 2;

            // Pastikan font Font Awesome sudah loaded dulu
            document.fonts.load('36px "Font Awesome 6 Free"').then(() => {
                ctx.save();
                ctx.translate(344, 390);

                emojis.forEach(emoji => {
                    const normalizedValue = (emoji.value - dataMin) / (dataMax - dataMin);
                    const totalGaugeAngle = 180;
                    const gaugeStartAngle = 120;
                    const emojiAngleDegrees = (normalizedValue * totalGaugeAngle) + gaugeStartAngle;
                    const angleRad = emojiAngleDegrees * Math.PI / 180;

                    const x = Math.cos(angleRad) * emojiRadius;
                    const y = Math.sin(angleRad) * emojiRadius;

                    ctx.fillStyle = 'white';
                    ctx.font = '36px "Font Awesome 6 Free"';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';

                    ctx.fillText(String.fromCharCode(parseInt(emoji.icon, 16)), x, y);
                });

                ctx.restore();
            });
        }
    };


    const storedResult = localStorage.getItem('analysisResult');

    if (storedResult) {
        const result = JSON.parse(storedResult);
        const percentage = parseFloat(result.hoaxPercentage) || 0;

        if (resultTitle) {
            resultTitle.textContent = `${result.originalText}`;
        }

        const fixedGaugeData = [20, 20, 20, 20, 20];

        if (resultDescription) {
            let message = '';
            if (percentage >= 80) {
                message = `<p class="description-text">Analisis menunjukkan indikasi hoax sebesar
                <span class="percentage" style="color: var(--x-danger-100);">${percentage.toFixed(2)}%</span>
                dan
                <span style="font-weight: bold; color:var(--x-danger-100);">masuk zona merah</span>. 
                Informasi ini terindikasi hoax.
                <span style="font-weight: bold; color:var(--x-danger-100);">Jangan langsung disebar ya!</span>
                </p>`;
            } else if (percentage >= 60) {
                message = `<p class="description-text">Analisis menunjukkan indikasi hoax sebesar
                <span class="percentage" style="color: var(--x-danger-60);">${percentage.toFixed(2)}%</span>
                dan
                <span style="font-weight: bold; color:var(--x-danger-60);">masuk zona merah</span>. 
                Informasi ini terindikasi hoax.
                <span style="font-weight: bold; color:var(--x-danger-60);">Jangan langsung disebar ya!</span>
                </p>`;
            } else if (percentage >= 40) {
                message = `<p class="description-text">Analisis menunjukkan indikasi hoax sebesar
                <span class="percentage" style="color: var(--x-warning-100);">${percentage.toFixed(2)}%</span>
                dan
                <span style="font-weight: bold; color:var(--x-warning-100);">masuk zona kuning</span>. 
                Kamu harus waspada dengan informasi ini.
                <span style="font-weight: bold; color:var(--x-warning-100);">Perlu diteliti lebih lanjut.</span>
                </p>`;
            } else if (percentage >= 20) {
                message = `<p class="description-text">Analisis menunjukkan indikasi hoax sebesar
                <span class="percentage" style="color: var(--x-safe-80);">${percentage.toFixed(2)}%</span>
                dan
                <span style="font-weight:400; color:var(--x-safe-80);">masuk zona hijau</span>. 
                Kamu bisa percaya pada informasi ini, tapi tetap cek sumber lain agar semakin valid ya!
                </p>`;
            } else {
                message = `<p class="description-text">Analisis menunjukkan indikasi hoax sebesar
                <span class="percentage" style="color: var(--x-safe-100);">${percentage.toFixed(2)}%</span>
                dan
                <span style="font-weight:400; color:var(--x-safe-100);">masuk zona hijau</span>. 
                Kamu bisa percaya pada informasi ini, tapi tetap cek sumber lain agar semakin valid ya!
                </p>`;
            }
            resultDescription.innerHTML = message;
        }

        // Render gauge statis
        const ctx = document.getElementById('gaugeChart');
        if (ctx) {
            const chart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: fixedGaugeData,
                        backgroundColor: [
                            '#06801D',
                            '#679E60',
                            '#FAD542',
                            '#FF6666',
                            '#FF0000'
                        ],
                        borderWidth: 0,
                    }, { // Dataset kedua (untuk ring abu-abu di tengah)
                        data: [100], // Satu segmen penuh
                        backgroundColor: '#D3D3D3', // Warna abu-abu terang
                        borderColor: '#D3D3D3', // Border juga abu-abu (opsional)
                        borderWidth: 0,
                        // Properti ini penting agar dataset kedua terlihat sebagai ring di dalam
                        weight: 0.4 // Menjadikan dataset ini lebih tipis dari dataset utama
                    }]
                },
                options: {
                    animation: false,
                    circumference: 270,
                    rotation: -134,
                    cutout: '52%',
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    }
                },
                plugins: [gaugeEmoji]
            });

            // Tambahkan overlay animasi jarum
            createNeedleOverlay(chart, percentage, 2000);
        }
    } else {
        window.location.href = 'home.html';
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            window.location.href = 'home.html';
        });
    }
});
