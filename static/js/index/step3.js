// ------------------------------------------------------------
// STEP 3: HISTORICAL RECORDS
// ------------------------------------------------------------
function populateHistoricalForm() {
    let suppOpts = searchBounds.supports.length ? searchBounds.supports : DEFAULT_CONFIG.supports;
    let a1Opts = searchBounds.amine1.length ? searchBounds.amine1 : DEFAULT_CONFIG.amine1;
    let a2Opts = searchBounds.amine2.length ? searchBounds.amine2 : DEFAULT_CONFIG.amine2;

    let sel = document.getElementById('exp-support');
    sel.innerHTML = '<option value="">Support</option>';
    suppOpts.forEach(s => sel.innerHTML += `<option value="${s}">${s}</option>`);
    sel = document.getElementById('exp-amine1');
    sel.innerHTML = '<option value="">Amine 1</option>';
    a1Opts.forEach(a => sel.innerHTML += `<option value="${a}">${a}</option>`);
    sel = document.getElementById('exp-amine2');
    sel.innerHTML = '<option value="No">No</option>';
    a2Opts.forEach(a => { if (a !== 'No') sel.innerHTML += `<option value="${a}">${a}</option>`; });
}

async function uploadHistoricalData() {
    const fileInput = document.getElementById('historicalFile');
    const file = fileInput.files[0];
    if (!file) {
        showNotification('Please select a CSV file', 'warning');
        return;
    }

    showLoading();
    const reader = new FileReader();
    reader.onload = async function(e) {
        const content = e.target.result;
        // Simple CSV parsing (assumes header row)
        const lines = content.split('\n');
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
        const records = [];

        for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
            const record = {};
            headers.forEach((h, idx) => {
                record[h] = values[idx];
            });
            // Map expected columns
            const mapped = {
                Support: record['Support'] || '',
                Amine_1_or_Additive_1: record['Amine_1_or_Additive_1'] || record['Amine1'] || '',
                Amine_2_or_Additive_2: record['Amine_2_or_Additive_2'] || record['Amine2'] || 'No',
                Organic_Content_pct: parseFloat(record['Organic_Content_pct'] || record['OC'] || 0),
                CO2_Capacity_mmol_g: parseFloat(record['CO2_Capacity_mmol_g'] || record['Capacity'] || 0)
            };
            if (mapped.Support && mapped.Amine_1_or_Additive_1 && !isNaN(mapped.Organic_Content_pct) && !isNaN(mapped.CO2_Capacity_mmol_g)) {
                records.push(mapped);
            }
        }

        // Add to historicalRecords
        historicalRecords = historicalRecords.concat(records);
        updateHistoricalRecordsTable();
        fileInput.value = '';
        showNotification(`Added ${records.length} records from CSV`, 'success');
        hideLoading();
    };
    reader.readAsText(file);
}

function addIndividualRecord() {
    let support = document.getElementById('exp-support').value;
    let amine1 = document.getElementById('exp-amine1').value;
    let amine2 = document.getElementById('exp-amine2').value;
    let oc = parseFloat(document.getElementById('exp-oc').value);
    let bet = parseFloat(document.getElementById('exp-bet').value);
    let pore = parseFloat(document.getElementById('exp-pore').value);
    let cap = parseFloat(document.getElementById('exp-capacity').value);
    if (!support || !amine1 || isNaN(oc) || isNaN(cap)) {
        showNotification('Please fill all required fields with valid numbers.', 'warning');
        return;
    }
    let rec = {
        Support: support,
        Amine_1_or_Additive_1: amine1,
        Amine_2_or_Additive_2: amine2,
        Organic_Content_pct: oc,
        BET_Bare_Surface_Area_m2_g: isNaN(bet) ? 0.0 : bet,
        Average_Bare_Pore_Diameter_nm: isNaN(pore) ? 0.0 : pore,
        CO2_Capacity_mmol_g: cap
    };
    historicalRecords.push(rec);
    updateHistoricalRecordsTable();
    // Clear form
    document.getElementById('exp-support').value = '';
    document.getElementById('exp-amine1').value = '';
    document.getElementById('exp-amine2').value = 'No';
    document.getElementById('exp-oc').value = '';
    document.getElementById('exp-bet').value = '';
    document.getElementById('exp-pore').value = '';
    document.getElementById('exp-capacity').value = '';
}

function updateHistoricalRecordsTable() {
    const tbody = document.getElementById('historicalRecordsTable');
    const countSpan = document.getElementById('recordCount');
    if (historicalRecords.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No records</td></tr>';
        countSpan.innerText = '0';
        return;
    }
    let html = '';
    historicalRecords.forEach((rec, idx) => {
        html += `<tr>
            <td>${rec.Support || ''}</td>
            <td>${rec.Amine_1_or_Additive_1 || ''}</td>
            <td>${rec.Amine_2_or_Additive_2 || ''}</td>
            <td>${rec.Organic_Content_pct ? rec.Organic_Content_pct.toFixed(1) : ''}</td>
            <td>${rec.BET_Bare_Surface_Area_m2_g ? rec.BET_Bare_Surface_Area_m2_g.toFixed(2) : ''}</td>
            <td>${rec.Average_Bare_Pore_Diameter_nm ? rec.Average_Bare_Pore_Diameter_nm.toFixed(2) : ''}</td>
            <td>${rec.CO2_Capacity_mmol_g ? rec.CO2_Capacity_mmol_g.toFixed(3) : ''}</td>
            <td><button class="btn btn-sm btn-outline-danger" onclick="removeHistoricalRecord(${idx})"><i class="fas fa-trash"></i></button></td>
        </tr>`;
    });
    tbody.innerHTML = html;
    countSpan.innerText = historicalRecords.length;
}

function removeHistoricalRecord(idx) {
    historicalRecords.splice(idx, 1);
    updateHistoricalRecordsTable();
}

async function initializeSystem() {
    if (historicalRecords.length === 0 && !confirm('No historical records. Continue anyway?')) {
        return;
    }
    showLoading();
    try {
        const payload = {
            searchBounds: searchBounds,
            conditions: conditions,
            historicalRecords: historicalRecords
        };
        const resp = await fetch('/api/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (data.success) {
            showNotification('System initialized successfully', 'success');
            goToStep(4);
        } else {
            showNotification(data.error || 'Initialization failed', 'error');
        }
    } catch (e) {
        showNotification(e.message, 'error');
    } finally {
        hideLoading();
    }
}

function submitHistoricalRecords() {
    initializeSystem(); // Alias
}
