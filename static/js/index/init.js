// ------------------------------------------------------------
// INITIALIZATION
// ------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function() {
    initMultiSelect();
    goToStep(1);

    // Set up slider listeners for Step 1
    document.getElementById('ocMinSlider').addEventListener('input', function() {
        let val = parseInt(this.value);
        let max = parseInt(document.getElementById('ocMaxSlider').value);
        if (val > max) { document.getElementById('ocMaxSlider').value = val; max = val; }
        searchBounds.ocRange = [val, max];
        document.getElementById('ocMinValue').innerText = val;
        document.getElementById('ocMaxValue').innerText = max;
        updateStep1Summary();
    });
    document.getElementById('ocMaxSlider').addEventListener('input', function() {
        let val = parseInt(this.value);
        let min = parseInt(document.getElementById('ocMinSlider').value);
        if (val < min) { document.getElementById('ocMinSlider').value = val; min = val; }
        searchBounds.ocRange = [min, val];
        document.getElementById('ocMinValue').innerText = min;
        document.getElementById('ocMaxValue').innerText = val;
        updateStep1Summary();
    });

    // Initialize condition radio listener
    document.querySelectorAll('input[name="testMethod"]').forEach(r => {
        r.addEventListener('change', updateStep2Summary);
    });
    updateStep2Summary();

    // Check URL parameters for direct navigation to step 4
    const urlParams = new URLSearchParams(window.location.search);
    const sessionParam = urlParams.get('session');
    const stepParam = urlParams.get('step');

    if (sessionParam) {
        // Show loading while setting up the session
        showLoading();

        // Set the session ID in the Flask session
        fetch('/api/set-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionParam })
        })
        .then(resp => resp.json())
        .then(data => {
            if (data.success) {
                // Load the session data into the UI
                return loadSystemStatus().then(() => {
                    // If step is specified, go to that step
                    if (stepParam) {
                        goToStep(parseInt(stepParam));
                        if (parseInt(stepParam) === 4) {
                            // Already loaded via loadSystemStatus()
                            updateOptimizationChart();
                        }
                    } else {
                        goToStep(4); // Default to step 4
                        updateOptimizationChart();
                    }
                });
            } else {
                showNotification('Failed to load session: ' + (data.error || 'Unknown error'), 'error');
                goToStep(1);
            }
        })
        .catch(err => {
            console.error('Error setting session:', err);
            showNotification('Error loading session', 'error');
            goToStep(1);
        })
        .finally(() => {
            hideLoading();
        });
    }
});
