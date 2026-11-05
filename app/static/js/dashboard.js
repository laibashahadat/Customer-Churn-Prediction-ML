async function loadDashboard() {
    const response = await fetch('/api/dashboard');
    const data = await response.json();
    document.getElementById('totalCustomers').textContent = data.total_customers || 0;
    document.getElementById('churnedCustomers').textContent = data.churned_customers || 0;
    document.getElementById('activeCustomers').textContent = data.active_customers || 0;
    document.getElementById('churnRate').textContent = ((data.churn_rate || 0) * 100).toFixed(1) + '%';
    document.getElementById('highRiskCustomers').textContent = data.high_risk_customers || 0;

    const distributionChart = new Chart(document.getElementById('distributionChart'), {
        type: 'doughnut',
        data: {
            labels: ['Churned', 'Active'],
            datasets: [{
                label: 'Churn Distribution',
                data: [data.distribution?.Yes || 0, data.distribution?.No || 0],
                backgroundColor: ['#ef4444', '#3b82f6']
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    const riskChart = new Chart(document.getElementById('riskChart'), {
        type: 'bar',
        data: {
            labels: ['Low', 'Medium', 'High'],
            datasets: [{
                label: 'Risk Category',
                data: [30, 25, 45],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444']
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

async function fetchModelPerformance() {
    const element = document.getElementById('performanceCard');
    if (!element) return;
    const response = await fetch('/api/model-performance');
    if (!response.ok) {
        element.innerHTML = '<div class="alert alert-warning">Model not yet trained.</div>';
        return;
    }
    const data = await response.json();
    const best = data.best_model || 'N/A';
    const metrics = data.results || {};
    const tableRows = Object.entries(metrics).map(([name, value]) => {
        return '<tr><td>' + name + '</td><td>' + (value.accuracy || 0).toFixed(3) + '</td><td>' + (value.precision || 0).toFixed(3) + '</td><td>' + (value.recall || 0).toFixed(3) + '</td><td>' + (value.f1 || 0).toFixed(3) + '</td><td>' + (value.roc_auc || 0).toFixed(3) + '</td></tr>';
    }).join('');
    element.innerHTML = '<h4>Best model: ' + best + '</h4><table class="table"><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>' + tableRows + '</tbody></table>';
}
