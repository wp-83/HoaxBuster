document.addEventListener('DOMContentLoaded', function() {
    const activityListContainer = document.getElementById('activity-list');
    const resetButton = document.getElementById('reset-local-storage'); // ambil tombol reset

    function getColorByPercentage(percentage) {
        if (percentage >= 80) {
            return 'var(--x-danger-100)';
        } else if (percentage >= 60) {
            return 'var(--x-danger-10)';
        } else if (percentage >= 40) {
            return 'var(--x-warning-10)';
        } else if (percentage >= 20) {
            return 'var(--x-safe-10)';
        } else {
            return 'var(--x-safe-100)';
        }
    }

    function getColorByPercentageText(percentage) {
        if (percentage >= 80) {
            return 'var(--x-danger-10)';
        } else if (percentage >= 60) {
            return 'var(--x-danger-100)';
        } else if (percentage >= 40) {
            return 'var(--x-warning-100)';
        } else if (percentage >= 20) {
            return 'var(--x-safe-100)';
        } else {
            return 'var(--x-safe-10)';
        }
    }

    function renderActivities() {
        const activities = JSON.parse(localStorage.getItem('searchHistory')) || [];
        activityListContainer.innerHTML = ''; // Kosongkan daftar sebelum render ulang

        if (activities.length === 0) {
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-state';
            emptyState.textContent = 'Belum ada aktivitas.';
            activityListContainer.appendChild(emptyState);
            return;
        }

        activities.forEach((activity, index) => {
            const itemElement = document.createElement('div');
            itemElement.className = 'content-frame';

            const percentage = parseFloat(activity.hoaxPercentage);
            const color = getColorByPercentage(percentage);
            const colorText = getColorByPercentageText(percentage);

            itemElement.innerHTML = `
                <div class="information-container">
                    <div class="original-text">${activity.originalText}</div>
                    <div class="right-side-info">
                        <div class="status-box" style="background-color: ${color};">
                            <p class="status-text" style="color: ${colorText};">${percentage.toFixed(2)}%</p>
                        </div>
                        <button class="delete-btn" data-index="${index}"><img src="../assets/icons/trash.svg" alt="trash" style="width: 1.5rem; height: 1.5rem;"></button>
                    </div>
                </div>
            `;

            activityListContainer.appendChild(itemElement);
        });

        // Tambahkan event listener untuk tombol hapus
        document.querySelectorAll('.delete-btn').forEach(button => {
            button.addEventListener('click', function() {
                const indexToDelete = this.getAttribute('data-index');
                deleteActivity(indexToDelete);
            });
        });
    }

    function deleteActivity(index) {
        let activities = JSON.parse(localStorage.getItem('searchHistory')) || [];
        if (index > -1 && index < activities.length) {
            activities.splice(index, 1);
            localStorage.setItem('searchHistory', JSON.stringify(activities));
            renderActivities(); // Render ulang daftar
        }
    }

    // === Tambahkan reset semua history ===
    if (resetButton) {
        resetButton.addEventListener('click', function() {
            localStorage.removeItem('searchHistory'); // hapus hanya key searchHistory
            renderActivities(); // render ulang biar muncul "Belum ada aktivitas"
        });
    }

    renderActivities();
});
