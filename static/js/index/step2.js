// ------------------------------------------------------------
// STEP 2: CONDITIONS
// ------------------------------------------------------------
function selectCondition(cond) {
    ['co2','temperature','humidity','flowRate'].forEach(c => {
        document.getElementById(c+'Controls').style.display = 'none';
        document.getElementById(c+'Card').classList.remove('selected');
    });
    if (selectedCondition === cond) {
        selectedCondition = null;
        return;
    }
    selectedCondition = cond;
    document.getElementById(cond+'Card').classList.add('selected');
    document.getElementById(cond+'Controls').style.display = 'block';
}

function updateCo2Value(v) {
    let val = parseFloat(v);
    document.getElementById('co2Value').innerText = val.toFixed(2);
    conditions.co2Concentration = val;
    updateStep2Summary();
}

function updateTemperatureValue(v) {
    let val = parseFloat(v);
    document.getElementById('temperatureValue').innerText = val.toFixed(1);
    conditions.temperature = val;
    updateStep2Summary();
}

function updateHumidityValue(v) {
    let val = parseInt(v);
    document.getElementById('humidityValue').innerText = val;
    conditions.humidity = val;
    updateStep2Summary();
}

function updateFlowRateValue(v) {
    let val = parseFloat(v);
    document.getElementById('flowRateValue').innerText = val.toFixed(1);
    conditions.flowRate = val;
    updateStep2Summary();
}

function updateStep2Summary() {
    document.getElementById('finalCo2').innerText = conditions.co2Concentration.toFixed(2) + ' vol%';
    document.getElementById('finalTemperature').innerText = conditions.temperature.toFixed(1) + ' °C';
    document.getElementById('finalHumidity').innerText = conditions.humidity + ' %';
    document.getElementById('finalFlowRate').innerText = conditions.flowRate.toFixed(1) + ' mL/min';
    let method = document.querySelector('input[name="testMethod"]:checked');
    if (method) {
        conditions.testMethod = method.value;
        document.getElementById('finalTestMethod').innerText = method.value;
    }
}
