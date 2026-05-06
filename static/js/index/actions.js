// ------------------------------------------------------------
// GLOBAL ACTIONS (export, restart, etc.)
// ------------------------------------------------------------
async function exportData() {
    const sid = await getCurrentSessionId();
    if (!sid) {
        showNotification('No active session to export', 'warning');
        return;
    }
    window.open(`/db/export/csv?session_id=${sid}`, '_blank');
}

async function getCurrentSessionId() {
    try {
        const resp = await fetch('/api/get-status');
        const data = await resp.json();
        if (data.success) {
            return data.session_id;
        }
    } catch (e) {
        console.error('Error getting session ID:', e);
    }
    return null;
}

async function restartSetup() {
    if (confirm('Restart the 4‑step process? This will reset the current session.')) {
        showLoading();
        try {
            await fetch('/api/reset-system', { method: 'POST' });
            showNotification('Session reset', 'info');
            // Clear historical records and go to step 1
            historicalRecords = [];
            currentCandidates = [];
            goToStep(1);
        } catch (e) {
            showNotification(e.message, 'error');
        } finally {
            hideLoading();
        }
    }
}
