// main.js
// Frontend JavaScript for Classroom Safety Narrator

let isStreaming = false;
let videoStream = null;
let processInterval = null;
let cameraIndex = 0;

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cameraSelect = document.getElementById('cameraSelect');
const videoStreamImg = document.getElementById('videoStream');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const stateBadge = document.getElementById('stateBadge');
const stateDescription = document.getElementById('stateDescription');
const narrationText = document.getElementById('narrationText');
const detectionsList = document.getElementById('detectionsList');
const detectionCount = document.getElementById('detectionCount');
const historyList = document.getElementById('historyList');
const historyCount = document.getElementById('historyCount');

// State colors mapping
const stateColors = {
    'normal class activity': '#4CAF50',
    'lone child may need attention': '#FF9800',
    'dangerous object near child': '#F44336',
    'empty classroom': '#9E9E9E'
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
});

function setupEventListeners() {
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    cameraSelect.addEventListener('change', (e) => {
        cameraIndex = parseInt(e.target.value);
    });
}

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        updateStatus('ready', 'System Ready');
    } catch (error) {
        updateStatus('error', 'Connection Error');
        console.error('Health check failed:', error);
    }
}

async function startCamera() {
    if (isStreaming) return;
    
    cameraIndex = parseInt(cameraSelect.value);
    
    try {
        updateStatus('active', 'Starting camera...');
        
        // Start camera on server
        const response = await fetch('/api/start_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ camera_index: cameraIndex })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Failed to start camera');
        }
        
        isStreaming = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        cameraSelect.disabled = true;
        
        // Show video stream
        videoPlaceholder.style.display = 'none';
        videoStreamImg.style.display = 'block';
        videoStreamImg.src = `/api/video_feed?t=${Date.now()}`;
        
        // Start processing frames
        startFrameProcessing();
        
        updateStatus('active', 'Camera Active');
        
    } catch (error) {
        updateStatus('error', `Error: ${error.message}`);
        alert(`Failed to start camera: ${error.message}\n\nTry a different camera index.`);
        console.error('Start camera error:', error);
    }
}

async function stopCamera() {
    if (!isStreaming) return;
    
    try {
        isStreaming = false;
        
        // Stop frame processing
        stopFrameProcessing();
        
        // Stop camera on server
        await fetch('/api/stop_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        // Hide video stream
        videoStreamImg.src = '';
        videoStreamImg.style.display = 'none';
        videoPlaceholder.style.display = 'flex';
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        cameraSelect.disabled = false;
        
        // Reset UI
        resetUI();
        updateStatus('ready', 'Ready');
        
    } catch (error) {
        console.error('Stop camera error:', error);
        updateStatus('error', 'Error stopping camera');
    }
}

function startFrameProcessing() {
    // Poll for current state every 1 second
    processInterval = setInterval(() => {
        fetchCurrentState();
        fetchHistory();
    }, 1000);
    // Also fetch history immediately
    fetchHistory();
}

function stopFrameProcessing() {
    if (processInterval) {
        clearInterval(processInterval);
        processInterval = null;
    }
}

async function fetchCurrentState() {
    if (!isStreaming) return;
    
    try {
        // Fetch current state from video feed processing
        const response = await fetch('/api/current_state');
        const data = await response.json();
        
        if (data.success) {
            // Update UI with results
            updateUI(data);
        }
        
    } catch (error) {
        console.error('Fetch state error:', error);
    }
}

function updateUI(data) {
    // Update state badge
    if (data.state) {
        stateBadge.textContent = data.state.toUpperCase();
        stateBadge.style.background = data.state_color || '#2196F3';
    }
    
    // Update state description
    if (data.narration) {
        stateDescription.textContent = data.narration;
    }
    
    // Update narration text - always show something
    if (data.narration && data.narration.trim()) {
        narrationText.textContent = data.narration;
        narrationText.style.display = 'block';
    } else if (data.state) {
        narrationText.textContent = `Scene analysis: ${data.state}.`;
        narrationText.style.display = 'block';
    } else {
        narrationText.textContent = 'Waiting for analysis...';
        narrationText.style.display = 'block';
    }
    
    // Update detections
    if (data.detections && data.detections.length > 0) {
        detectionCount.textContent = data.detections.length;
        detectionsList.innerHTML = '';
        
        data.detections.forEach(det => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            
            // Use textContent to prevent XSS
            const className = document.createElement('span');
            className.className = 'class-name';
            className.textContent = det.class;
            
            const confidence = document.createElement('span');
            confidence.className = 'confidence';
            confidence.textContent = `${(det.confidence * 100).toFixed(1)}%`;
            
            item.appendChild(className);
            item.appendChild(confidence);
            detectionsList.appendChild(item);
        });
    } else {
        detectionCount.textContent = '0';
        detectionsList.innerHTML = '<p class="empty-state">No objects detected</p>';
    }
}

async function fetchHistory() {
    if (!isStreaming) return;
    
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.success && data.history) {
            updateHistoryUI(data.history);
        }
    } catch (error) {
        console.error('Fetch history error:', error);
    }
}

function updateHistoryUI(history) {
    if (!history || history.length === 0) {
        historyCount.textContent = '0';
        historyList.innerHTML = '<p class="empty-state">No history yet</p>';
        return;
    }
    
    historyCount.textContent = history.length;
    historyList.innerHTML = '';
    
    // Display up to 20 most recent entries
    const displayHistory = history.slice(0, 20);
    
    displayHistory.forEach(entry => {
        const item = document.createElement('div');
        item.className = 'history-item';
        
        const timestamp = new Date(entry.timestamp * 1000);
        const timeStr = timestamp.toLocaleTimeString();
        
        // Create elements safely to prevent XSS
        const header = document.createElement('div');
        header.className = 'history-header';
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'history-time';
        timeSpan.textContent = timeStr;
        
        const stateBadge = document.createElement('span');
        stateBadge.className = 'history-state-badge';
        stateBadge.style.background = entry.state_color || '#2196F3';
        stateBadge.style.color = 'white';
        stateBadge.style.padding = '4px 8px';
        stateBadge.style.borderRadius = '4px';
        stateBadge.style.fontSize = '0.85em';
        stateBadge.style.fontWeight = '600';
        stateBadge.textContent = entry.state.toUpperCase();
        
        header.appendChild(timeSpan);
        header.appendChild(stateBadge);
        
        const narrationDiv = document.createElement('div');
        narrationDiv.className = 'history-narration';
        narrationDiv.textContent = entry.narration || 'No narration';
        
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'history-details';
        
        const detectionsSpan = document.createElement('span');
        detectionsSpan.className = 'history-detections';
        detectionsSpan.textContent = `${entry.detections_count || 0} objects detected`;
        
        detailsDiv.appendChild(detectionsSpan);
        
        item.appendChild(header);
        item.appendChild(narrationDiv);
        item.appendChild(detailsDiv);
        
        historyList.appendChild(item);
    });
}

function resetUI() {
    stateBadge.textContent = 'Waiting...';
    stateBadge.style.background = '#9E9E9E';
    stateDescription.textContent = 'No analysis yet';
    narrationText.textContent = 'Waiting for analysis...';
    narrationText.style.display = 'block';
    detectionCount.textContent = '0';
    detectionsList.innerHTML = '<p class="empty-state">No objects detected</p>';
    historyCount.textContent = '0';
    historyList.innerHTML = '<p class="empty-state">No history yet</p>';
}

function updateStatus(type, message) {
    statusText.textContent = message;
    statusDot.className = 'status-dot';
    
    if (type === 'active') {
        statusDot.classList.add('active');
    } else if (type === 'error') {
        statusDot.classList.add('error');
    }
}

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (isStreaming) {
        fetch('/api/stop_camera', { method: 'POST' }).catch(() => {});
    }
});

