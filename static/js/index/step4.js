// ------------------------------------------------------------
// STEP 4: OPTIMIZATION
// ------------------------------------------------------------
async function loadSystemStatus() {
    try {
        const resp = await fetch('/api/get-status');
        if (!resp.ok) {
            // No active session – maybe stay in step 3
            return;
        }
        const data = await resp.json();
        if (data.success) {
            document.getElementById('modelAccuracy').innerText = data.best_capacity.toFixed(4);
            document.getElementById('iterationCount').innerText = data.iteration;
            document.getElementById('realExperiments').innerText = data.total_experiments;
            document.getElementById('dataPoints').innerText = data.total_data_points;
            const histCount = data.historical_records || (data.total_data_points - data.total_experiments);
            document.getElementById('dataPoints').title = `Historical: ${histCount} + Real: ${data.total_experiments}`;

            // Update conditions display
            const conditions = data.conditions || {};
            document.getElementById('displayTemperature').innerText = `${conditions.temperature?.toFixed(1) || 'N/A'}°C`;
            document.getElementById('displayCO2').innerText = `${conditions.co2Concentration?.toFixed(2) || 'N/A'} vol%`;
            document.getElementById('displayHumidity').innerText = `${conditions.humidity !== undefined && conditions.humidity !== null ? conditions.humidity : 'N/A'}%`;
            document.getElementById('displayFlowRate').innerText = `${conditions.flowRate?.toFixed(1) || 'N/A'} mL/min`;
            document.getElementById('displayTestMethod').innerText = conditions.testMethod || 'N/A';

            // Update search bounds display AND store globally for use in other functions
            const searchBounds = data.search_bounds || {};

            // Store in global variables for use in other functions
            window.searchBounds = searchBounds;
            window.conditions = conditions;

            const ocRange = searchBounds.ocRange || [0, 100];
            document.getElementById('displayOCRange').innerText = `${ocRange[0]} - ${ocRange[1]}%`;

            // Display support materials with overflow fix
            const supports = searchBounds.supports || [];
            const supportsElement = document.getElementById('displaySupports');
            if (supports.length > 0) {
                // Limit to first 3 items and show "+X more" if there are more
                if (supports.length > 3) {
                    const shownItems = supports.slice(0, 3).map(s => `<span class="badge bg-secondary me-1">${s}</span>`).join('');
                    supportsElement.innerHTML = `${shownItems}<span class="badge bg-info ms-1">+${supports.length - 3} more</span>`;
                } else {
                    supportsElement.innerHTML = supports.map(s => `<span class="badge bg-secondary me-1">${s}</span>`).join('');
                }
            } else {
                supportsElement.innerHTML = '<span class="badge bg-secondary">None selected</span>';
            }

            // Display amine 1 materials with overflow fix
            const amine1 = searchBounds.amine1 || [];
            const amine1Element = document.getElementById('displayAmine1');
            if (amine1.length > 0) {
                // Limit to first 3 items and show "+X more" if there are more
                if (amine1.length > 3) {
                    const shownItems = amine1.slice(0, 3).map(a => `<span class="badge bg-secondary me-1">${a}</span>`).join('');
                    amine1Element.innerHTML = `${shownItems}<span class="badge bg-info ms-1">+${amine1.length - 3} more</span>`;
                } else {
                    amine1Element.innerHTML = amine1.map(a => `<span class="badge bg-secondary me-1">${a}</span>`).join('');
                }
            } else {
                amine1Element.innerHTML = '<span class="badge bg-secondary">None selected</span>';
            }

            // Display amine 2 materials - show all with wrap
            const amine2 = searchBounds.amine2 || [];
            const amine2Element = document.getElementById('displayAmine2');
            if (amine2.length > 0) {
                amine2Element.innerHTML = amine2.map(a => `<span class="badge bg-secondary me-1 mb-1">${a}</span>`).join('');
            } else {
                amine2Element.innerHTML = '<span class="badge bg-secondary">None selected</span>';
            }

            // Display additive 3 materials - show all with wrap

            // Display support-specific ranges for only selected supports with multi-column layout
            const selectedSupports = searchBounds.supports || [];
            const supportSpecificRanges = searchBounds.supportSpecificRanges || {};
            const rangesContainer = document.getElementById('displaySupportSpecificRanges');

            if (selectedSupports.length > 0) {
                let rangesHtml = '';

                // Count how many selected supports have specific ranges defined
                let supportsWithRanges = 0;
                for (const support of selectedSupports) {
                    if (supportSpecificRanges[support]) {
                        supportsWithRanges++;

                        const ranges = supportSpecificRanges[support];
                        const betRange = ranges.betRange || [0, 1000];
                        const poreRange = ranges.poreRange || [0, 20];

                        // Create a column for each support with its ranges
                        rangesHtml += `
                            <div class="col-md-6 col-lg-4 mb-2">
                                <div class="card">
                                    <div class="card-header bg-light py-2">
                                        <strong class="text-primary">${support}</strong>
                                    </div>
                                    <div class="card-body py-2">
                                        <div><strong>BET:</strong> ${betRange[0]}-${betRange[1]} m²/g</div>
                                        <div><strong>Pore:</strong> ${poreRange[0]}-${poreRange[1]} nm</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                }

                if (rangesHtml !== '') {
                    rangesContainer.innerHTML = rangesHtml;
                } else {
                    rangesContainer.innerHTML = '<div class="col-12"><em>No support-specific ranges defined for selected supports</em></div>';
                }
            } else {
                rangesContainer.innerHTML = '<div class="col-12"><em>No supports selected</em></div>';
            }

            if (data.best_experiment) {
                let best = data.best_experiment;
                let cap = data.best_capacity;
                document.getElementById('bestExperimentCard').style.display = 'block';
                document.getElementById('bestExperimentContent').innerHTML = `
                    <div class="text-center">
                        <h4 class="text-success">${cap.toFixed(4)} mmol/g</h4>
                        <hr>
                        <div><strong>Support:</strong> ${best.Support || 'N/A'}</div>
                        <div><strong>Amine 1:</strong> ${best.Amine_1_or_Additive_1 || 'N/A'}</div>
                        <div><strong>Amine 2:</strong> ${best.Amine_2_or_Additive_2 || 'N/A'}</div>
                        <div><strong>Org:</strong> ${best.Organic_Content_pct ? best.Organic_Content_pct.toFixed(1) : 'N/A'}%</div>
                        <div><strong>BET:</strong> ${best.BET_Bare_Surface_Area_m2_g ? best.BET_Bare_Surface_Area_m2_g.toFixed(2) : 'N/A'} m²/g</div>
                        <div><strong>Pore:</strong> ${best.Average_Bare_Pore_Diameter_nm ? best.Average_Bare_Pore_Diameter_nm.toFixed(2) : 'N/A'} nm</div>
                    </div>
                `;
            } else {
                document.getElementById('bestExperimentCard').style.display = 'none';
            }

            // Load recent experiments
            await loadRecentExperiments(data.session_id);
            // Load current candidates if any
            const sessionResp = await fetch(`/db/session/${data.session_id}`);
            const sessionData = await sessionResp.json();
            if (sessionData.success && sessionData.session.current_candidates) {
                currentCandidates = sessionData.session.current_candidates;
                displayCandidates(currentCandidates);
            }
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadRecentExperiments(sessionId) {
    try {
        const resp = await fetch(`/db/experiments?session_id=${sessionId}`);
        const data = await resp.json();
        if (data.success) {
            const recent = data.experiments.slice(-5).reverse();
            const list = document.getElementById('recentExperimentsList');
            if (recent.length === 0) {
                list.innerHTML = 'No experiments yet';
                return;
            }
            let html = '<div class="list-group">';
            recent.forEach((exp, index) => {
                const cand = exp.candidate || {};
                html += `
                    <div class="list-group-item exp-list-item" data-exp-index="${index}" data-exp-data='${JSON.stringify(exp).replace(/'/g, '&apos;')}'>
                        <div class="d-flex justify-content-between">
                            <small>${cand.Support || ''}|${cand.Amine_1_or_Additive_1 || ''}|${cand.Amine_2_or_Additive_2 || ''}|${cand.Organic_Content_pct ? cand.Organic_Content_pct.toFixed(1) : ''}%</small>
                            <span class="badge bg-success">${exp.experimental_performance.toFixed(3)}</span>
                        </div>
                        <small class="text-muted">${new Date(exp.timestamp).toLocaleString()}</small>
                    </div>
                `;
            });
            html += '</div>';
            list.innerHTML = html;

            // Add event listeners to the experiment list items
            document.querySelectorAll('.exp-list-item').forEach(item => {
                item.addEventListener('click', function() {
                    const expData = JSON.parse(this.getAttribute('data-exp-data'));
                    showExperimentDetails(this.getAttribute('data-exp-index'), expData);
                });
            });
        }
    } catch (e) {
        console.error(e);
    }
}

async function updateOptimizationChart() {
    try {
        const resp = await fetch('/api/generate-chart');
        const data = await resp.json();
        if (data.success && data.chart_data) {
            document.getElementById('optimizationChartImg').src = data.chart_data;
        }
    } catch (e) {
        console.error(e);
    }
}

async function generateCandidates() {
    showLoading();
    try {
        const resp = await fetch('/api/generate-candidates', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_candidates: 5 })
        });
        const data = await resp.json();
        if (data.success) {
            currentCandidates = data.candidates;
            displayCandidates(data.candidates);
            // Check if model uncertainty is high and consult LLM
            if (typeof checkUncertaintyAndSuggest === 'function') {
                checkUncertaintyAndSuggest(data.candidates);
            }
            showNotification(`Generated ${data.candidates.length} candidates`, 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (e) {
        showNotification(e.message, 'error');
    } finally {
        hideLoading();
    }
}

function displayCandidates(candidates) {
    let container = document.getElementById('candidatesContainer');
    if (!candidates || candidates.length === 0) {
        container.innerHTML = '<div class="text-center text-muted p-4">No candidates</div>';
        document.getElementById('candidateCount').innerText = '0';
        // Update the global variable as well
        window.currentCandidates = [];
        return;
    }
    let html = '<div class="list-group">';
    // Assign candidates to the global variable to ensure they're accessible
    window.currentCandidates = candidates;
    candidates.forEach((c, idx) => {
        html += `
            <div class="list-group-item candidate-card">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">#${idx+1} – ${c.Predicted_CO2_Capacity_mmol_g.toFixed(4)} mmol/g</h6>
                    <small class="text-info">Uncertainty: ${c.Uncertainty}</small>
                </div>
                <p class="mb-1 small">
                    Support: ${c.Support} · Amine1: ${c.Amine_1_or_Additive_1} · Amine2: ${c.Amine_2_or_Additive_2} ·
                    Org: ${c.Organic_Content_pct.toFixed(1)}% ·
                    BET: ${c.BET_Bare_Surface_Area_m2_g ? c.BET_Bare_Surface_Area_m2_g.toFixed(2) : 'N/A'} ·
                    Pore: ${c.Average_Bare_Pore_Diameter_nm ? c.Average_Bare_Pore_Diameter_nm.toFixed(2) : 'N/A'}
                </p>
                <button class="btn btn-sm btn-outline-success" onclick="prepareRecordModalForCandidate(${idx})">
                    <i class="fas fa-flask me-1"></i>Record
                </button>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
    document.getElementById('candidateCount').innerText = candidates.length;
}

// Function to show detailed view of an experiment when clicked
function showExperimentDetails(index, experiment) {
    // Create a modal to display experiment details
    let modalHtml = `
        <div class="modal fade" id="experimentDetailModal" tabindex="-1" aria-labelledby="experimentDetailLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="experimentDetailLabel">Experiment Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Material Properties</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Support:</strong> ${experiment.candidate?.Support || 'N/A'}</li>
                                    <li><strong>Amine 1:</strong> ${experiment.candidate?.Amine_1_or_Additive_1 || 'N/A'}</li>
                                    <li><strong>Amine 2:</strong> ${experiment.candidate?.Amine_2_or_Additive_2 || 'N/A'}</li>
                                    <li><strong>Organic Content (%):</strong> ${experiment.candidate?.Organic_Content_pct ? experiment.candidate.Organic_Content_pct.toFixed(2) : 'N/A'}</li>
                                    <li><strong>BET Surface Area (m²/g):</strong> ${experiment.candidate?.BET_Bare_Surface_Area_m2_g ? experiment.candidate.BET_Bare_Surface_Area_m2_g.toFixed(2) : 'N/A'}</li>
                                    <li><strong>Pore Diameter (nm):</strong> ${experiment.candidate?.Average_Bare_Pore_Diameter_nm ? experiment.candidate.Average_Bare_Pore_Diameter_nm.toFixed(2) : 'N/A'}</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6>Experimental Conditions</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Temperature (°C):</strong> ${experiment.Temperature || experiment.candidate?.Temperature || 'N/A'}</li>
                                    <li><strong>CO₂ Conc (vol%):</strong> ${experiment.CO2_Concentration || experiment.candidate?.CO2_Concentration || 'N/A'}</li>
                                    <li><strong>Humidity (%):</strong> ${experiment.Humidity || experiment.candidate?.Humidity || 'N/A'}</li>
                                    <li><strong>Flow Rate (mL/min):</strong> ${experiment.Flow_Rate || experiment.candidate?.Flow_Rate || 'N/A'}</li>
                                    <li><strong>Test Method:</strong> ${experiment.Test_Method || experiment.candidate?.Test_Method || 'N/A'}</li>
                                </ul>

                                <h6 class="mt-3">Performance</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Experimental Capacity:</strong> ${experiment.experimental_performance ? experiment.experimental_performance.toFixed(4) : 'N/A'} mmol/g</li>
                                    <li><strong>Predicted Capacity:</strong> ${experiment.predicted_performance ? experiment.predicted_performance.toFixed(4) : 'N/A'} mmol/g</li>
                                    <li><strong>Uncertainty:</strong> ${experiment.uncertainty ? experiment.uncertainty : 'N/A'}</li>
                                    <li><strong>Is Historical:</strong> ${experiment.is_historical ? 'Yes' : 'No'}</li>
                                    <li><strong>Timestamp:</strong> ${experiment.timestamp ? new Date(experiment.timestamp).toLocaleString() : 'N/A'}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove any existing modal first
    const existingModal = document.getElementById('experimentDetailModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add the modal to the body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('experimentDetailModal'));
    modal.show();
}
