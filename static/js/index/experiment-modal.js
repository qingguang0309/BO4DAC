// ------------------------------------------------------------
// EXPERIMENT MODAL & RECORDING
// ------------------------------------------------------------
function openRecordModal(idx) {
    // Set up the form with the candidate's values pre-filled
    const cand = (window.currentCandidates && window.currentCandidates[idx]) ? window.currentCandidates[idx] : null;

    if (!cand) {
        showNotification('Candidate data not available', 'warning');
        return;
    }

    // Pre-populate the form fields with candidate values
    document.getElementById('exp-support-modal').value = cand.Support || '';
    document.getElementById('exp-amine1-modal').value = cand.Amine_1_or_Additive_1 || '';
    document.getElementById('exp-amine2-modal').value = cand.Amine_2_or_Additive_2 || 'No';
    document.getElementById('exp-oc-modal').value = cand.Organic_Content_pct || '';
    document.getElementById('exp-bet-modal').value = cand.BET_Bare_Surface_Area_m2_g || '';
    document.getElementById('exp-pore-modal').value = cand.Average_Bare_Pore_Diameter_nm || '';
    document.getElementById('exp-capacity-modal').value = cand.Predicted_CO2_Capacity_mmol_g.toFixed(3) || '';

    // Set conditions from candidate or global conditions
    document.getElementById('exp-temp-modal').value = cand.Temperature || (window.conditions && window.conditions.temperature) || 25.0;
    document.getElementById('exp-co2-modal').value = cand.CO2_Concentration || (window.conditions && window.conditions.co2Concentration) || 0.04;
    document.getElementById('exp-humidity-modal').value = cand.Humidity !== undefined ? cand.Humidity : (window.conditions && window.conditions.humidity) || 0;
    document.getElementById('exp-flow-modal').value = cand.Flow_Rate || (window.conditions && window.conditions.flowRate) || 100.0;
    document.getElementById('exp-test-method-modal').value = cand.Test_Method || (window.conditions && window.conditions.testMethod) || 'TGA';

    // Store the original predicted capacity and candidate to preserve it when submitting
    window.originalPredictedCapacity = cand.Predicted_CO2_Capacity_mmol_g || 0.0;
    window.currentCandidateIdx = idx;
    window.currentCandidates = window.currentCandidates || []; // Ensure it exists

    // Show the comprehensive modal
    const modal = new bootstrap.Modal(document.getElementById('recordExperimentModal'));
    modal.show();
}

// Function to prepare the modal specifically for a candidate from the list
function prepareRecordModalForCandidate(idx) {
    // Additional check to ensure dropdowns have values if searchBounds is not available
    ensureDropdownHasOptions();
    // Ensure we're using the global currentCandidates array
    if (!window.currentCandidates || window.currentCandidates.length <= idx) {
        showNotification('Candidate data not available', 'warning');
        return;
    }
    openRecordModal(idx);
}

// This function is deprecated - keeping for compatibility but shouldn't be used
// The preferred method is submitFullExperiment called from submitExperimentFromModal
async function submitExperiment(idx) {
    let cand = (window.currentCandidates && window.currentCandidates[idx]) ? window.currentCandidates[idx] : null;
    if (!cand) {
        showNotification('Candidate data not available', 'warning');
        return;
    }
    let actual = parseFloat(document.getElementById('actualCapacity').value);
    if (isNaN(actual)) {
        showNotification('Please enter a valid number', 'warning');
        return;
    }
    console.info(cand);
    showLoading();
    try {
        // Use the comprehensive API endpoint that preserves the original predicted capacity
        const candidate = {
            'Support': cand.Support,
            'Amine_1_or_Additive_1': cand.Amine_1_or_Additive_1,
            'Amine_2_or_Additive_2': cand.Amine_2_or_Additive_2,
            'Organic_Content_pct': cand.Organic_Content_pct,
            'BET_Bare_Surface_Area_m2_g': cand.BET_Bare_Surface_Area_m2_g,
            'Average_Bare_Pore_Diameter_nm': cand.Average_Bare_Pore_Diameter_nm,
            'Predicted_CO2_Capacity_mmol_g': cand.Predicted_CO2_Capacity_mmol_g,
            'Uncertainty': cand.Uncertainty
        };

        const resp = await fetch('/api/record-experiment-full', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                candidate: candidate,
                actual_capacity: actual,
                original_predicted_capacity: cand.Predicted_CO2_Capacity_mmol_g,
                notes: ''
            })
        });
        const data = await resp.json();
        if (data.success) {
            showNotification(`Recorded: Measured=${actual.toFixed(4)}, Predicted=${cand.Predicted_CO2_Capacity_mmol_g.toFixed(4)} mmol/g`, 'success');
            bootstrap.Modal.getInstance(document.getElementById('recordExperimentModal')).hide();
            await loadSystemStatus();
            await updateOptimizationChart();
            // Optionally refresh candidates
            if (window.currentCandidates && window.currentCandidates.length) {
                displayCandidates(window.currentCandidates);
            }
        } else {
            showNotification(data.error, 'error');
        }
    } catch (e) {
        showNotification(e.message, 'error');
    } finally {
        hideLoading();
    }
}

function openRecordExperimentModal() {
    // Initialize the form with current session's available options
    populateRecordExperimentForm();

    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('recordExperimentModal'));
    modal.show();

    // Additional check to ensure dropdowns have values if searchBounds is not available
    ensureDropdownHasOptions();
}

function ensureDropdownHasOptions() {
    // If searchBounds is not available or empty, use default config options
    const supports = (window.searchBounds && window.searchBounds.supports && window.searchBounds.supports.length > 0)
        ? window.searchBounds.supports
        : DEFAULT_CONFIG.supports;
    const amine1 = (window.searchBounds && window.searchBounds.amine1 && window.searchBounds.amine1.length > 0)
        ? window.searchBounds.amine1
        : DEFAULT_CONFIG.amine1;
    const amine2 = (window.searchBounds && window.searchBounds.amine2 && window.searchBounds.amine2.length > 0)
        ? window.searchBounds.amine2
        : DEFAULT_CONFIG.amine2;

    // Populate support dropdown
    const supportSelect = document.getElementById('exp-support-modal');
    if (supportSelect && supportSelect.options.length <= 1) { // Only has default option
        supports.forEach(s => {
            if (!Array.from(supportSelect.options).some(opt => opt.value === s)) {
                const option = document.createElement('option');
                option.value = s;
                option.textContent = s;
                supportSelect.appendChild(option);
            }
        });
    }

    // Populate amine1 dropdown
    const amine1Select = document.getElementById('exp-amine1-modal');
    if (amine1Select && amine1Select.options.length <= 1) { // Only has default option
        amine1.forEach(a => {
            if (!Array.from(amine1Select.options).some(opt => opt.value === a)) {
                const option = document.createElement('option');
                option.value = a;
                option.textContent = a;
                amine1Select.appendChild(option);
            }
        });
    }

    // Populate amine2 dropdown
    const amine2Select = document.getElementById('exp-amine2-modal');
    if (amine2Select && amine2Select.options.length <= 1) { // Only has default option
        amine2.forEach(a => {
            if (a !== 'No' && !Array.from(amine2Select.options).some(opt => opt.value === a)) {
                const option = document.createElement('option');
                option.value = a;
                option.textContent = a;
                amine2Select.appendChild(option);
            }
        });
    }
}

function populateRecordExperimentForm() {
    // Get available options from current session bounds, falling back to default config
    const supports = (window.searchBounds && window.searchBounds.supports && window.searchBounds.supports.length > 0)
        ? window.searchBounds.supports
        : DEFAULT_CONFIG.supports;
    const amine1 = (window.searchBounds && window.searchBounds.amine1 && window.searchBounds.amine1.length > 0)
        ? window.searchBounds.amine1
        : DEFAULT_CONFIG.amine1;
    const amine2 = (window.searchBounds && window.searchBounds.amine2 && window.searchBounds.amine2.length > 0)
        ? window.searchBounds.amine2
        : DEFAULT_CONFIG.amine2;

    // Populate support dropdown
    const supportSelect = document.getElementById('exp-support-modal');
    supportSelect.innerHTML = '<option value="">Select Support</option>';
    supports.forEach(s => {
        const option = document.createElement('option');
        option.value = s;
        option.textContent = s;
        supportSelect.appendChild(option);
    });

    // Populate amine1 dropdown
    const amine1Select = document.getElementById('exp-amine1-modal');
    amine1Select.innerHTML = '<option value="">Select Amine 1</option>';
    amine1.forEach(a => {
        const option = document.createElement('option');
        option.value = a;
        option.textContent = a;
        amine1Select.appendChild(option);
    });

    // Populate amine2 dropdown
    const amine2Select = document.getElementById('exp-amine2-modal');
    amine2Select.innerHTML = '<option value="No">No</option>';
    amine2.forEach(a => {
        if (a !== 'No') {
            const option = document.createElement('option');
            option.value = a;
            option.textContent = a;
            amine2Select.appendChild(option);
        }
    });

    // Set default values based on current conditions if available
    if (window.conditions) {
        document.getElementById('exp-temp-modal').value = conditions.temperature || 25.0;
        document.getElementById('exp-co2-modal').value = conditions.co2Concentration || 0.04;
        document.getElementById('exp-humidity-modal').value = conditions.humidity || 0;
        document.getElementById('exp-flow-modal').value = conditions.flowRate || 100.0;
        document.getElementById('exp-test-method-modal').value = conditions.testMethod || 'TGA';
    }
}

function submitExperimentFromModal() {
    // Get values from form
    const supportEl = document.getElementById('exp-support-modal');
    const amine1El = document.getElementById('exp-amine1-modal');
    const amine2El = document.getElementById('exp-amine2-modal');
    const ocEl = document.getElementById('exp-oc-modal');
    const betEl = document.getElementById('exp-bet-modal');
    const poreEl = document.getElementById('exp-pore-modal');
    const capacityEl = document.getElementById('exp-capacity-modal');

    const candidate = {
        'Support': supportEl ? supportEl.value : '',
        'Amine_1_or_Additive_1': amine1El ? amine1El.value : '',
        'Amine_2_or_Additive_2': amine2El ? amine2El.value : 'No',
        'Organic_Content_pct': ocEl && ocEl.value ? parseFloat(ocEl.value) : 0,
        'BET_Bare_Surface_Area_m2_g': betEl && betEl.value ? parseFloat(betEl.value) : 0,
        'Average_Bare_Pore_Diameter_nm': poreEl && poreEl.value ? parseFloat(poreEl.value) : 0
    };

    const capacity = capacityEl && capacityEl.value ? parseFloat(capacityEl.value) : NaN;
    const notes = document.getElementById('exp-notes-modal') ? document.getElementById('exp-notes-modal').value : '';

    if (!candidate.Support || !candidate['Amine_1_or_Additive_1'] ||
        isNaN(candidate['Organic_Content_pct']) ||
        isNaN(candidate['BET_Bare_Surface_Area_m2_g']) ||
        isNaN(candidate['Average_Bare_Pore_Diameter_nm']) ||
        isNaN(capacity)) {
        showNotification('Please fill all required fields with valid values.', 'warning');
        return;
    }

    // Get current conditions from form
    const tempEl = document.getElementById('exp-temp-modal');
    const co2El = document.getElementById('exp-co2-modal');
    const humidityEl = document.getElementById('exp-humidity-modal');
    const flowEl = document.getElementById('exp-flow-modal');
    const testMethodEl = document.getElementById('exp-test-method-modal');

    const experimental_conditions = {
        'Temperature': tempEl && tempEl.value ? parseFloat(tempEl.value) : 25.0,
        'CO2_Concentration': co2El && co2El.value ? parseFloat(co2El.value) : 0.04,
        'Humidity': humidityEl && humidityEl.value ? parseFloat(humidityEl.value) : 0,
        'Flow_Rate': flowEl && flowEl.value ? parseFloat(flowEl.value) : 100.0,
        'Test_Method': (testMethodEl && testMethodEl.value) ? testMethodEl.value : ((window.conditions && window.conditions.testMethod) || 'TGA'),
        'Notes': notes
    };

    // Add conditions to the candidate object
    Object.assign(candidate, experimental_conditions);

    // Submit the experiment
    submitFullExperiment(candidate, capacity);
}

async function submitFullExperiment(candidate, capacity) {
    showLoading();
    try {
        // Include the uncertainty from the original candidate if available
        if (window.currentCandidateIdx !== undefined && window.currentCandidates &&
            window.currentCandidateIdx < window.currentCandidates.length) {
            const originalCandidate = window.currentCandidates[window.currentCandidateIdx];
            candidate.Uncertainty = originalCandidate.Uncertainty || 0.0;
            candidate.Predicted_CO2_Capacity_mmol_g = originalCandidate.Predicted_CO2_Capacity_mmol_g || 0.0;
        }

        const resp = await fetch('/api/record-experiment-full', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                candidate: candidate,
                actual_capacity: capacity,
                original_predicted_capacity: window.originalPredictedCapacity || 0.0
            })
        });
        const data = await resp.json();
        if (data.success) {
            showNotification(`Recorded: Measured=${capacity.toFixed(4)}, Predicted=${(window.originalPredictedCapacity || 0.0).toFixed(4)} mmol/g`, 'success');
            bootstrap.Modal.getInstance(document.getElementById('recordExperimentModal')).hide();

            // Clear the original predicted capacity value
            window.originalPredictedCapacity = undefined;
            window.currentCandidateIdx = undefined;

            // Refresh the optimization status to show updated information
            await loadSystemStatus();
            updateOptimizationChart();
        } else {
            showNotification(data.error || 'Failed to record experiment', 'error');
        }
    } catch (e) {
        showNotification(e.message, 'error');
    } finally {
        hideLoading();
    }
}

function recordExperiment() {
    if (!currentCandidates.length) {
        showNotification('Generate candidates first', 'warning');
        return;
    }
    openRecordModal(0);
}
