// ------------------------------------------------------------
// UI Utilities
// ------------------------------------------------------------
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}
function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}
function showNotification(msg, type = 'info') {
    const el = document.getElementById('notification');
    el.className = `notification ${type}`;
    el.innerText = msg;
    el.style.display = 'block';
    setTimeout(() => {
        el.style.display = 'none';
    }, 3000);
}
