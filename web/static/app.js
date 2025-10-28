
// Connect to SocketIO server
const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

// Chart configurations
const maxDataPoints = 50;
const cpuData = { labels: [], values: [] };
const latencyData = { labels: [], values: [] };

let frameCount = 0;
let latencySum = 0;

// Initialize CPU chart
const cpuChart = new Chart(document.getElementById('cpuChart'), {
    type: 'line',
    data: {
        labels: cpuData.labels,
        datasets: [{
            label: 'CPU %',
            data: cpuData.values,
            borderColor: 'rgb(220, 53, 69)',
            backgroundColor: 'rgba(220, 53, 69, 0.1)',
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { beginAtZero: true, max: 100 }
        },
        animation: { duration: 0 }
    }
});

// Initialize Latency chart
const latencyChart = new Chart(document.getElementById('latencyChart'), {
    type: 'line',
    data: {
        labels: latencyData.labels,
        datasets: [{
            label: 'Latency (ms)',
            data: latencyData.values,
            borderColor: 'rgb(255, 193, 7)',
            backgroundColor: 'rgba(255, 193, 7, 0.1)',
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { beginAtZero: true }
        },
        animation: { duration: 0 }
    }
});

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    document.getElementById('status').textContent = '● 已连接';
    document.getElementById('status').className = 'badge bg-success status-badge';
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    document.getElementById('status').textContent = '● 断开连接';
    document.getElementById('status').className = 'badge bg-danger status-badge';
});

socket.on('system_status', (data) => {
    console.log('System status:', data);
    document.getElementById('modelPath').textContent = data.model;
    document.getElementById('threads').textContent = data.threads;
});

socket.on('inference_result', (data) => {
    // Update metrics
    document.getElementById('topClass').textContent = truncateText(data.top_class, 20);
    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('latency').textContent = data.latency_ms.toFixed(2);
    document.getElementById('cpuUsage').textContent = data.cpu_percent.toFixed(1) + '%';
    
    // Update frame count and average latency
    frameCount++;
    latencySum += data.latency_ms;
    document.getElementById('frameCount').textContent = frameCount;
    document.getElementById('avgLatency').textContent = (latencySum / frameCount).toFixed(2);
    
    // Update Top-5 table
    const tbody = document.querySelector('#topKTable tbody');
    tbody.innerHTML = '';
    data.top_k.forEach((item, idx) => {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = idx + 1;
        row.insertCell(1).textContent = truncateText(item.class, 30);
        row.insertCell(2).textContent = (item.prob * 100).toFixed(2) + '%';
    });
    
    // Update charts
    const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
    
    updateChart(cpuData, cpuChart, timestamp, data.cpu_percent);
    updateChart(latencyData, latencyChart, timestamp, data.latency_ms);
});

// Helper functions
function updateChart(dataObj, chartObj, label, value) {
    dataObj.labels.push(label);
    dataObj.values.push(value);
    
    if (dataObj.labels.length > maxDataPoints) {
        dataObj.labels.shift();
        dataObj.values.shift();
    }
    
    chartObj.update();
}

function truncateText(text, maxLen) {
    return text.length > maxLen ? text.substring(0, maxLen) + '...' : text;
}
