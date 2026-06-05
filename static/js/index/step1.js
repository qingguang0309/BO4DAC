// ------------------------------------------------------------
// STEP 1: SEARCH BOUNDS
// ------------------------------------------------------------
function initMultiSelect() {
    // Supports
    const supportsContainer = document.getElementById('supportsContainer');
    supportsContainer.innerHTML = '';
    DEFAULT_CONFIG.supports.forEach(s => {
        const div = document.createElement('div');
        div.className = `multi-select-item ${searchBounds.supports.includes(s) ? 'selected' : ''}`;
        div.textContent = s;
        div.onclick = () => toggleSelection('supports', s, div);
        supportsContainer.appendChild(div);
    });

    // Amine 1
    const amine1Container = document.getElementById('amine1Container');
    amine1Container.innerHTML = '';
    DEFAULT_CONFIG.amine1.forEach(a => {
        const div = document.createElement('div');
        div.className = `multi-select-item ${searchBounds.amine1.includes(a) ? 'selected' : ''}`;
        div.textContent = a;
        div.onclick = () => toggleSelection('amine1', a, div);
        amine1Container.appendChild(div);
    });

    // Amine 2
    const amine2Container = document.getElementById('amine2Container');
    amine2Container.innerHTML = '';
    DEFAULT_CONFIG.amine2.forEach(a => {
        const div = document.createElement('div');
        div.className = `multi-select-item ${searchBounds.amine2.includes(a) ? 'selected' : ''}`;
        div.textContent = a;
        div.onclick = () => toggleSelection('amine2', a, div);
        amine2Container.appendChild(div);
    });


    // OC Sliders
    document.getElementById('ocMinSlider').value = searchBounds.ocRange[0];
    document.getElementById('ocMaxSlider').value = searchBounds.ocRange[1];
    document.getElementById('ocMinValue').innerText = searchBounds.ocRange[0];
    document.getElementById('ocMaxValue').innerText = searchBounds.ocRange[1];
}

function toggleSelection(type, value, el) {
    let arr = searchBounds[type];
    if (arr.includes(value)) {
        arr = arr.filter(v => v !== value);
        el.classList.remove('selected');

        // If deselecting a support, also remove its specific ranges
        if (type === 'supports' && searchBounds.supportSpecificRanges) {
            delete searchBounds.supportSpecificRanges[value];
        }
    } else {
        arr.push(value);
        el.classList.add('selected');

        // If selecting a support for the first time, initialize its ranges
        if (type === 'supports') {
            if (!searchBounds.supportSpecificRanges) {
                searchBounds.supportSpecificRanges = {};
            }

            if (!searchBounds.supportSpecificRanges[value]) {
                // Set default ranges based on support type
                if (value.includes('SBA') || value.toLowerCase().includes('sba')) {
                    searchBounds.supportSpecificRanges[value] = {
                        betRange: [300, 1000],
                        poreRange: [5, 15]
                    };
                } else if (value.includes('MCM') || value.toLowerCase().includes('mcm')) {
                    searchBounds.supportSpecificRanges[value] = {
                        betRange: [800, 1200],
                        poreRange: [2, 5]
                    };
                } else {
                    // Default ranges
                    searchBounds.supportSpecificRanges[value] = {
                        betRange: [300, 1000],
                        poreRange: [2, 15]
                    };
                }
            }
        }
    }
    searchBounds[type] = arr;
    updateStep1Summary();
}

function toggleBadge(type, value, el) {
    toggleSelection(type, value, el);
}

function selectAll(type) {
    searchBounds[type] = [...DEFAULT_CONFIG[type]];
    initMultiSelect();
}

function deselectAll(type) {
    searchBounds[type] = [];
    initMultiSelect();
}

function updateStep1Summary() {
    document.getElementById('summarySupports').innerText = searchBounds.supports.length + ' selected';
    document.getElementById('summaryAmine1').innerText = searchBounds.amine1.length + ' selected';
    document.getElementById('summaryAmine2').innerText = searchBounds.amine2.length + ' selected';
    document.getElementById('summaryOCRange').innerText = searchBounds.ocRange[0] + ' - ' + searchBounds.ocRange[1] + ' %';

    // Update support-specific ranges display
    updateSupportSpecificRangesDisplay();
}

function updateSupportSpecificRangesDisplay() {
    const supports = searchBounds.supports || [];
    const container = document.getElementById('supportRangesContainer');
    const card = document.getElementById('supportSpecificRangesCard');

    if (supports.length === 0) {
        card.style.display = 'none';
        return;
    }

    card.style.display = 'block';
    let html = '';

    supports.forEach(support => {
        // Get current ranges for this support or set defaults
        if (!searchBounds.supportSpecificRanges) {
            searchBounds.supportSpecificRanges = {};
        }

        if (!searchBounds.supportSpecificRanges[support]) {
            // Set default ranges based on support type
            if (support.includes('SBA') || support.toLowerCase().includes('sba')) {
                searchBounds.supportSpecificRanges[support] = {
                    betRange: [300, 1000],
                    poreRange: [5, 15]
                };
            } else if (support.includes('MCM') || support.toLowerCase().includes('mcm')) {
                searchBounds.supportSpecificRanges[support] = {
                    betRange: [800, 1200],
                    poreRange: [2, 5]
                };
            } else {
                // Default ranges
                searchBounds.supportSpecificRanges[support] = {
                    betRange: [300, 1000],
                    poreRange: [2, 15]
                };
            }
        }

        const ranges = searchBounds.supportSpecificRanges[support];
        const betMin = ranges.betRange[0];
        const betMax = ranges.betRange[1];
        const poreMin = ranges.poreRange[0];
        const poreMax = ranges.poreRange[1];

        html += `
            <div class="row mb-3 p-2 border rounded bg-light">
                <div class="col-md-6">
                    <h6>${support}</h6>
                </div>
                <div class="col-md-6 text-end">
                    <button class="btn btn-sm btn-outline-danger" onclick="removeSupport('${support}')">
                        <i class="fas fa-times"></i> Remove
                    </button>
                </div>
                <div class="col-md-6 mt-2">
                    <label class="form-label">BET Surface Area Range (m²/g)</label>
                    <div class="d-flex align-items-center">
                        <input type="number" class="form-control form-control-sm me-2"
                               value="${betMin}"
                               onchange="updateSupportSpecificRange('${support}', 'bet', 'min', this.value)"
                               style="max-width: 100px;">
                        <span class="mx-1">-</span>
                        <input type="number" class="form-control form-control-sm ms-2"
                               value="${betMax}"
                               onchange="updateSupportSpecificRange('${support}', 'bet', 'max', this.value)"
                               style="max-width: 100px;">
                    </div>
                </div>
                <div class="col-md-6 mt-2">
                    <label class="form-label">Pore Diameter Range (nm)</label>
                    <div class="d-flex align-items-center">
                        <input type="number" class="form-control form-control-sm me-2"
                               value="${poreMin}"
                               onchange="updateSupportSpecificRange('${support}', 'pore', 'min', this.value)"
                               style="max-width: 100px;" step="0.1">
                        <span class="mx-1">-</span>
                        <input type="number" class="form-control form-control-sm ms-2"
                               value="${poreMax}"
                               onchange="updateSupportSpecificRange('${support}', 'pore', 'max', this.value)"
                               style="max-width: 100px;" step="0.1">
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function updateSupportSpecificRange(support, param, minMax, value) {
    if (!searchBounds.supportSpecificRanges) {
        searchBounds.supportSpecificRanges = {};
    }

    if (!searchBounds.supportSpecificRanges[support]) {
        searchBounds.supportSpecificRanges[support] = {
            betRange: [300, 1000],
            poreRange: [2, 15]
        };
    }

    const ranges = searchBounds.supportSpecificRanges[support];

    if (param === 'bet') {
        if (minMax === 'min') {
            ranges.betRange[0] = parseFloat(value);
        } else {
            ranges.betRange[1] = parseFloat(value);
        }
        // Ensure min <= max
        if (ranges.betRange[0] > ranges.betRange[1]) {
            if (minMax === 'min') {
                ranges.betRange[1] = ranges.betRange[0];
                // Update the max input field
                const maxInput = document.querySelector(`input[value="${parseFloat(value)}"]:nth-of-type(2)[onchange*="'${support}'"]`);
                if (maxInput && maxInput.previousElementSibling && maxInput.previousElementSibling.value === value) {
                    // Find the correct max input for this support
                    const allInputs = document.querySelectorAll(`input[onchange*="'${support}'"]`);
                    for (let input of allInputs) {
                        if (input.onchange.toString().includes("updateSupportRange") &&
                            input.onchange.toString().includes("'bet', 'max'")) {
                            input.value = ranges.betRange[1];
                            break;
                        }
                    }
                }
            } else {
                ranges.betRange[0] = ranges.betRange[1];
            }
        }
    } else if (param === 'pore') {
        if (minMax === 'min') {
            ranges.poreRange[0] = parseFloat(value);
        } else {
            ranges.poreRange[1] = parseFloat(value);
        }
        // Ensure min <= max
        if (ranges.poreRange[0] > ranges.poreRange[1]) {
            if (minMax === 'min') {
                ranges.poreRange[1] = ranges.poreRange[0];
            } else {
                ranges.poreRange[0] = ranges.poreRange[1];
            }
        }
    }

    searchBounds.supportSpecificRanges[support] = ranges;
}

function removeSupport(support) {
    // Remove from supports list
    searchBounds.supports = searchBounds.supports.filter(s => s !== support);

    // Remove from support-specific ranges
    if (searchBounds.supportSpecificRanges) {
        delete searchBounds.supportSpecificRanges[support];
    }

    // Update UI
    initMultiSelect(); // This will update the multi-select UI
    updateStep1Summary();
}

async function addCustomMaterial(category) {
    const inputMap = { supports: 'addSupportInput', amine1: 'addAmine1Input', amine2: 'addAmine2Input' };
    const input = document.getElementById(inputMap[category]);
    const name = (input.value || '').trim();
    if (!name) return;

    // Add to DEFAULT_CONFIG so it appears in the list
    if (!DEFAULT_CONFIG[category].includes(name)) {
        DEFAULT_CONFIG[category].push(name);
    }

    // Auto-select the new material
    if (!searchBounds[category].includes(name)) {
        searchBounds[category].push(name);
    }

    // Persist to backend if session exists
    try {
        await fetch('/api/add-material', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category, name })
        });
    } catch (e) {
        console.warn('Failed to sync custom material to backend:', e);
    }

    input.value = '';
    initMultiSelect();
    updateStep1Summary();

    // Flash the new item
    const containers = { supports: 'supportsContainer', amine1: 'amine1Container', amine2: 'amine2Container' };
    const container = document.getElementById(containers[category]);
    const items = container.querySelectorAll('.multi-select-item');
    items.forEach(item => {
        if (item.textContent === name) {
            item.style.transition = 'background-color 0.3s';
            item.style.backgroundColor = '#d4edda';
            setTimeout(() => { item.style.backgroundColor = ''; }, 1500);
        }
    });
}
