// ------------------------------------------------------------
// GLOBAL CONFIGURATION & STATE
// ------------------------------------------------------------
const DEFAULT_CONFIG = {
    supports: [
        'SBA-15', 'NS', 'MCM-41', 'MCM-48', 'Mesoporous γ-Al2O3', 'MMSN',
        'MCF', 'MMON', 'MPS', 'BHMS', 'SA', 'FAU', 'MIL-101(Cr)', 'MCM-36',
        'THMS', 'MF', 'NPREXAD4', 'NPRED4020', 'PREXAD7', 'PREHP2MG',
        'PREDA201', 'NPREHP20', 'R-CFA-SBA-15', 'W-CFA-SBA-15', 'ZN', 'AC',
        'FS', 'CNS', 'CA', 'PREHPD450', 'MPC'
    ],
    amine1: [
        'BPEI', 'TEPA', 'DEA', 'DETA', 'LPEI', 'Ph-3-ED', 'Ph-3-PD',
        'Ph-6-ED', 'Ph-6-PD', 'PEG200', 'TETA', 'TPTA', 'EI-Den', 'PI-Den',
        'AM-TEPA', 'PAA', 'GPAA', 'CTMA+', 'PPG', 'LPPI', 'PGA', 'PZ',
        'MEA', 'EDA', 'Spermine', 'Spermidine', 'TREN', 'EP', 'EB-TEPA',
        'PEHA', 'AN-TEPA'
    ],
    amine2: [
        'No', 'DEA', 'CTAB', 'P123', 'PC', 'PEG200', 'SDS', 'Span80',
        'PEG1000', 'CTAC', 'DPPD', 'TBD', 'DBPD', 'BHT', 'PET', 'TDE',
        'HEDS', 'DTDP', 'BTES', 'APTES', 'TEOT', 'CTMA+'
    ],
};

let currentStep = 1;
let searchBounds = {
    supports: [DEFAULT_CONFIG.supports[0]],
    amine1: [DEFAULT_CONFIG.amine1[0]],
    amine2: [DEFAULT_CONFIG.amine2[0]],
    ocRange: [0, 100]
};
let conditions = {
    co2Concentration: 0.04,
    temperature: 25.0,
    humidity: 0,
    flowRate: 100.0,
    testMethod: 'TGA'
};

window.searchBounds = searchBounds;
window.conditions = conditions;
let historicalRecords = [];
let currentCandidates = [];
let selectedCondition = null;
