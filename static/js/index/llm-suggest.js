// ------------------------------------------------------------
// LLM-ASSISTED SUGGESTIONS (SSE Streaming)
// ------------------------------------------------------------
let llmUncertaintyThreshold = 0.5;
let llmSearchSources = [];  // populated from search_source SSE event

function checkUncertaintyAndSuggest(candidates) {
    const panel = document.getElementById('llmSuggestPanel');
    if (!panel) return;

    if (!candidates || candidates.length === 0) {
        panel.style.display = 'none';
        return;
    }

    // Compute average uncertainty
    const uncertainties = candidates.map(c => parseFloat(c.Uncertainty || 0));
    const avgUnc = uncertainties.reduce((a, b) => a + b, 0) / uncertainties.length;
    const threshold = llmUncertaintyThreshold;

    // Update display
    const avgUncEl = document.getElementById('llmAvgUncertainty');
    const threshEl = document.getElementById('llmThresholdDisplay');
    if (avgUncEl) avgUncEl.textContent = avgUnc.toFixed(4);
    if (threshEl) threshEl.textContent = threshold.toFixed(2);

    if (avgUnc >= threshold) {
        panel.style.display = 'block';
        document.getElementById('llmStatusBadge').innerHTML =
            '<span class="badge bg-warning text-dark"><i class="fas fa-exclamation-triangle me-1"></i>High Uncertainty — AI Consultation Recommended</span>';
        requestLLMSuggestions();
    } else {
        panel.style.display = 'block';
        document.getElementById('llmStatusBadge').innerHTML =
            '<span class="badge bg-success"><i class="fas fa-check-circle me-1"></i>Low Uncertainty — Model is Confident</span>';
        document.getElementById('llmSuggestionsContainer').innerHTML =
            '<div class="text-center text-muted p-3">Model uncertainty is low (confident). Click "Get AI Suggestions" if you want alternative perspectives.</div>';
    }
}

async function requestLLMSuggestions() {
    const container = document.getElementById('llmSuggestionsContainer');
    const liveOutput = document.getElementById('llmLiveOutput');
    const livePanel = document.getElementById('llmLivePanel');
    const suggestionsPanel = document.getElementById('llmSuggestionsPanel');

    // Show live streaming panel
    if (livePanel) livePanel.style.display = 'block';
    if (suggestionsPanel) suggestionsPanel.style.display = 'none';
    if (liveOutput) liveOutput.textContent = '';

    // Reset previous search sources and thinking
    llmSearchSources = [];
    const prevSourcesPanel = document.getElementById('llmSearchSourcePanel');
    if (prevSourcesPanel) prevSourcesPanel.style.display = 'none';
    const thinkPanel = document.getElementById('llmThinkingPanel');
    if (thinkPanel) thinkPanel.style.display = 'none';
    const thinkOutput = document.getElementById('llmThinkingOutput');
    if (thinkOutput) thinkOutput.textContent = '';

    // Disable button during request
    const btn = document.getElementById('llmSuggestBtn');
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Thinking...'; }

    try {
        const resp = await fetch('/api/llm-suggest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ threshold: llmUncertaintyThreshold })
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let meta = null;
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split('\n');
            // Keep the last incomplete line in the buffer
            buffer = lines.pop() || '';

            let currentEvent = null;
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (currentEvent === 'meta') {
                            meta = data;
                            const avgUncEl = document.getElementById('llmAvgUncertainty');
                            if (avgUncEl && data.avg_uncertainty !== undefined) {
                                avgUncEl.textContent = data.avg_uncertainty.toFixed(4);
                            }
                        } else if (currentEvent === 'search_source') {
                            // Store sources and display citation panel
                            llmSearchSources = data.sources || [];
                            const sourcesPanel = document.getElementById('llmSearchSourcePanel');
                            const sourcesList = document.getElementById('llmSearchSourceList');
                            if (sourcesPanel && sourcesList && llmSearchSources.length > 0) {
                                sourcesPanel.style.display = 'block';
                                let srcHtml = `<div class="d-flex align-items-center mb-2">
                                    <i class="fas fa-search text-primary me-2"></i>
                                    <span class="fw-semibold small">Referenced Sources</span>
                                    <span class="badge bg-primary ms-2">${llmSearchSources.length}</span>
                                </div>`;
                                srcHtml += '<ol class="mb-0 ps-3" style="font-size: 0.82em;">';
                                llmSearchSources.forEach(s => {
                                    srcHtml += `<li class="mb-1">
                                        <a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" class="text-decoration-none">
                                            ${escapeHtml(s.title)} <i class="fas fa-external-link-alt fa-xs ms-1"></i>
                                        </a>
                                    </li>`;
                                });
                                srcHtml += '</ol>';
                                sourcesList.innerHTML = srcHtml;
                            }
                        } else if (currentEvent === 'token') {
                            fullText += data.text || '';
                            if (liveOutput) {
                                liveOutput.textContent = fullText;
                                // Auto-scroll
                                liveOutput.scrollTop = liveOutput.scrollHeight;
                            }
                        } else if (currentEvent === 'thinking') {
                            // Show thinking/reasoning tokens in a separate muted area
                            // Note: thinking tokens are NOT added to fullText — they're chain-of-thought, not structured output
                            const thinkEl = document.getElementById('llmThinkingOutput');
                            if (thinkEl) {
                                thinkEl.textContent += data.text || '';
                                thinkEl.scrollTop = thinkEl.scrollHeight;
                                document.getElementById('llmThinkingPanel').style.display = 'block';
                            }
                        } else if (currentEvent === 'done') {
                            const suggestions = data.suggestions || [];
                            window.llmSuggestions = suggestions;
                            displayLLMSuggestions(suggestions);
                        } else if (currentEvent === 'error') {
                            container.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-circle me-1"></i>API Error: ${escapeHtml(data.error)}</div>`;
                        }
                    } catch (e) {
                        console.warn('Failed to parse SSE data:', line, e);
                    }
                    currentEvent = null;
                }
            }
        }

        // If no structured 'done' event was received, try to parse the accumulated text
        if (fullText && !window.llmSuggestions) {
            // Try JSON parse on accumulated text
            try {
                let cleaned = fullText.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
                const match = cleaned.match(/\[.*\]/s);
                if (match) {
                    const suggestions = JSON.parse(match[0]);
                    window.llmSuggestions = suggestions;
                    displayLLMSuggestions(suggestions);
                }
            } catch (e) {
                // Show raw text as fallback
                displayLLMSuggestions([{ reasoning: fullText, raw_response: true }]);
            }
        }

    } catch (e) {
        container.innerHTML = `<div class="alert alert-danger">Request failed: ${escapeHtml(e.message)}</div>`;
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-magic me-1"></i>Get AI Suggestions'; }
    }
}

function displayLLMSuggestions(suggestions) {
    const suggestionsPanel = document.getElementById('llmSuggestionsPanel');
    const suggestionsList = document.getElementById('llmSuggestionsList');

    if (!suggestions || suggestions.length === 0) {
        if (suggestionsPanel) suggestionsPanel.style.display = 'block';
        if (suggestionsList) suggestionsList.innerHTML = '<div class="text-center text-muted p-3">No structured suggestions parsed from AI response.</div>';
        return;
    }

    if (suggestionsPanel) suggestionsPanel.style.display = 'block';

    let html = '<div class="list-group">';
    suggestions.forEach((s, idx) => {
        if (s.raw_response) {
            html += `
                <div class="list-group-item">
                    <div class="small text-muted">AI Response (unstructured)</div>
                    <div class="mt-1" style="white-space: pre-wrap; max-height: 200px; overflow-y: auto;">${renderCitedReasoning(s.reasoning || '')}</div>
                </div>
            `;
            return;
        }

        const isNovel = !isDuplicateSuggestion(s);
        const noveltyBadge = isNovel
            ? '<span class="badge bg-info ms-2">Novel</span>'
            : '<span class="badge bg-secondary ms-2">Similar to tested</span>';

        html += `
            <div class="list-group-item llm-suggestion-card">
                <div class="d-flex w-100 justify-content-between align-items-start">
                    <h6 class="mb-1">
                        <i class="fas fa-lightbulb text-warning me-1"></i>
                        Suggestion #${idx + 1}${noveltyBadge}
                    </h6>
                    <button class="btn btn-sm btn-outline-primary" onclick="useLLMSuggestion(${idx})" title="Use this suggestion to record an experiment">
                        <i class="fas fa-flask me-1"></i>Try This
                    </button>
                </div>
                <p class="mb-1 small">
                    <strong>Support:</strong> ${s.Support || 'N/A'} &middot;
                    <strong>Amine1:</strong> ${s.Amine_1_or_Additive_1 || 'N/A'} &middot;
                    <strong>Amine2:</strong> ${s.Amine_2_or_Additive_2 || 'N/A'} &middot;
                    <strong>OC:</strong> ${s.Organic_Content_pct != null ? parseFloat(s.Organic_Content_pct).toFixed(1) : 'N/A'}% &middot;
                    <strong>BET:</strong> ${s.BET_Bare_Surface_Area_m2_g != null ? parseFloat(s.BET_Bare_Surface_Area_m2_g).toFixed(1) : 'N/A'} m&sup2;/g &middot;
                    <strong>Pore:</strong> ${s.Average_Bare_Pore_Diameter_nm != null ? parseFloat(s.Average_Bare_Pore_Diameter_nm).toFixed(2) : 'N/A'} nm
                </p>
                <div class="mt-1 p-2 bg-light rounded" style="font-size: 0.85em; max-height: 120px; overflow-y: auto;">
                    <i class="fas fa-brain text-purple me-1"></i><strong>AI Reasoning:</strong> ${renderCitedReasoning(s.reasoning || 'No reasoning provided')}
                </div>
            </div>
        `;
    });
    html += '</div>';

    if (suggestionsList) suggestionsList.innerHTML = html;
}

function isDuplicateSuggestion(suggestion) {
    if (!window.currentCandidates) return false;
    const s = suggestion;
    return window.currentCandidates.some(c =>
        c.Support === s.Support &&
        c.Amine_1_or_Additive_1 === s.Amine_1_or_Additive_1 &&
        c.Amine_2_or_Additive_2 === s.Amine_2_or_Additive_2
    );
}

function useLLMSuggestion(idx) {
    const suggestions = window.llmSuggestions || [];
    if (idx >= suggestions.length) return;

    const s = suggestions[idx];
    ensureDropdownHasOptions();

    const supportEl = document.getElementById('exp-support-modal');
    const amine1El = document.getElementById('exp-amine1-modal');
    const amine2El = document.getElementById('exp-amine2-modal');
    const ocEl = document.getElementById('exp-oc-modal');
    const betEl = document.getElementById('exp-bet-modal');
    const poreEl = document.getElementById('exp-pore-modal');
    const capacityEl = document.getElementById('exp-capacity-modal');

    if (supportEl) supportEl.value = s.Support || '';
    if (amine1El) amine1El.value = s.Amine_1_or_Additive_1 || '';
    if (amine2El) amine2El.value = s.Amine_2_or_Additive_2 || 'No';
    if (ocEl) ocEl.value = s.Organic_Content_pct || '';
    if (betEl) betEl.value = s.BET_Bare_Surface_Area_m2_g || '';
    if (poreEl) poreEl.value = s.Average_Bare_Pore_Diameter_nm || '';
    if (capacityEl) capacityEl.value = '';

    // Set conditions from suggestion or global session conditions
    const tempEl = document.getElementById('exp-temp-modal');
    const co2El = document.getElementById('exp-co2-modal');
    const humidityEl = document.getElementById('exp-humidity-modal');
    const flowEl = document.getElementById('exp-flow-modal');
    const testMethodEl = document.getElementById('exp-test-method-modal');
    if (tempEl) tempEl.value = s.Temperature || (window.conditions && window.conditions.temperature) || 25.0;
    if (co2El) co2El.value = s.CO2_Concentration || (window.conditions && window.conditions.co2Concentration) || 0.04;
    if (humidityEl) humidityEl.value = s.Humidity !== undefined ? s.Humidity : (window.conditions && window.conditions.humidity) || 0;
    if (flowEl) flowEl.value = s.Flow_Rate || (window.conditions && window.conditions.flowRate) || 100.0;
    if (testMethodEl) testMethodEl.value = s.Test_Method || (window.conditions && window.conditions.testMethod) || 'TGA';

    window.originalPredictedCapacity = undefined;
    window.currentCandidateIdx = undefined;

    const modal = new bootstrap.Modal(document.getElementById('recordExperimentModal'));
    modal.show();
}

function updateLLMThreshold(value) {
    llmUncertaintyThreshold = parseFloat(value);
    const display = document.getElementById('llmThresholdDisplay');
    if (display) display.textContent = llmUncertaintyThreshold.toFixed(2);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderCitedReasoning(text) {
    // First escape, then convert [N] markers into clickable superscript links
    let html = escapeHtml(text || '');
    if (llmSearchSources.length > 0) {
        html = html.replace(/\[(\d+)\]/g, (match, num) => {
            const idx = parseInt(num);
            const src = llmSearchSources.find(s => s.index === idx);
            if (src) {
                return `<a href="${escapeHtml(src.url)}" target="_blank" rel="noopener noreferrer"
                    class="citation-link" title="${escapeHtml(src.title)}"
                    style="color: #667eea; text-decoration: none; font-weight: 600; vertical-align: super; font-size: 0.75em;">[${num}]</a>`;
            }
            return match;
        });
    }
    return html;
}
