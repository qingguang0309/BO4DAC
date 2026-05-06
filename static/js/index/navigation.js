// ------------------------------------------------------------
// STEP NAVIGATION
// ------------------------------------------------------------
function goToStep(step) {
    currentStep = step;
    // Update step circles
    document.querySelectorAll('.step').forEach((el, idx) => {
        const stepNum = idx + 1;
        el.classList.remove('active', 'completed');
        if (stepNum < step) el.classList.add('completed');
        else if (stepNum === step) el.classList.add('active');
    });
    // Show correct content
    document.querySelectorAll('.step-content').forEach((el, idx) => {
        if (idx + 1 === step) el.classList.add('active');
        else el.classList.remove('active');
    });
    // Trigger per-step updates
    if (step === 1) updateStep1Summary();
    if (step === 2) updateStep2Summary();
    if (step === 3) { populateHistoricalForm(); updateHistoricalRecordsTable(); }
    if (step === 4) { loadSystemStatus(); updateOptimizationChart(); }

    // Update global variables for use in other functions
    window.searchBounds = searchBounds;
    window.conditions = conditions;
}
