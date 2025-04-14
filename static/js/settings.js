document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const ngramToggle = document.getElementById('ngram-toggle');
    const thresholdContainer = document.getElementById('threshold-container');
    const probabilityThreshold = document.getElementById('probability-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const saveButton = document.getElementById('save-settings');
    const resetButton = document.getElementById('reset-settings');

    // Load settings from cookies
    function loadSettings() {
        let settings = {
            useNgramApproximation: false,
            probabilityThreshold: 0.5
        };

        // Try to get settings from cookie
        const cookieValue = getCookie('cacheSettings');
        if (cookieValue) {
            try {
                settings = JSON.parse(cookieValue);
            } catch (e) {
                console.error('Error parsing settings cookie:', e);
            }
        }

        ngramToggle.checked = settings.useNgramApproximation;
        probabilityThreshold.value = settings.probabilityThreshold;
        thresholdValue.textContent = settings.probabilityThreshold;

        // Show/hide threshold container based on toggle
        thresholdContainer.style.display = settings.useNgramApproximation ? 'flex' : 'none';

        return settings;
    }

    // Helper function to get cookie value
    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
        return null;
    }

    // Save settings to cookies
    function saveSettings() {
        const settings = {
            useNgramApproximation: ngramToggle.checked,
            probabilityThreshold: parseFloat(probabilityThreshold.value)
        };

        // Save to cookie (expires in 30 days)
        const settingsJson = JSON.stringify(settings);
        const expiryDate = new Date();
        expiryDate.setDate(expiryDate.getDate() + 30);
        document.cookie = `cacheSettings=${settingsJson}; expires=${expiryDate.toUTCString()}; path=/`;

        // Show success message
        showMessage('Settings saved successfully!', 'success');
    }

    // Reset settings to defaults
    function resetSettings() {
        const defaultSettings = {
            useNgramApproximation: false,
            probabilityThreshold: 0.5
        };

        ngramToggle.checked = defaultSettings.useNgramApproximation;
        probabilityThreshold.value = defaultSettings.probabilityThreshold;
        thresholdValue.textContent = defaultSettings.probabilityThreshold;
        thresholdContainer.style.display = 'none';

        // Save default settings to cookie
        const settingsJson = JSON.stringify(defaultSettings);
        const expiryDate = new Date();
        expiryDate.setDate(expiryDate.getDate() + 30);
        document.cookie = `cacheSettings=${settingsJson}; expires=${expiryDate.toUTCString()}; path=/`;

        // Show success message
        showMessage('Settings reset to defaults.', 'success');
    }

    // Show message
    function showMessage(message, type) {
        // Check if message container exists
        let messageContainer = document.querySelector('.settings-message');

        // If not, create it
        if (!messageContainer) {
            messageContainer = document.createElement('div');
            messageContainer.className = 'settings-message';
            document.querySelector('.settings-actions').insertAdjacentElement('beforebegin', messageContainer);
        }

        // Set message content and class
        messageContainer.textContent = message;
        messageContainer.className = `settings-message ${type}`;

        // Hide message after 3 seconds
        setTimeout(() => {
            messageContainer.style.opacity = '0';
            setTimeout(() => {
                messageContainer.remove();
            }, 300);
        }, 3000);
    }

    // Add event listeners
    ngramToggle.addEventListener('change', function() {
        thresholdContainer.style.display = this.checked ? 'flex' : 'none';
    });

    probabilityThreshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
    });

    saveButton.addEventListener('click', saveSettings);
    resetButton.addEventListener('click', resetSettings);

    // Initialize settings
    loadSettings();
});
