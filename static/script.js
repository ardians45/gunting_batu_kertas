// --- Elemen HTML ---
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const resultElement = document.getElementById('result');
const chartCanvas = document.getElementById('predictionChart');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const recordingIndicator = document.getElementById('recordingIndicator');

// --- Konfigurasi ---
const PREDICTION_INTERVAL = 1500;
const LABELS = ['Kertas', 'Batu', 'Gunting'];

let isCameraOn = false;
let isPredicting = false;
let stream = null;

// --- Inisialisasi Chart.js ---
const predictionChart = new Chart(chartCanvas, {
  type: 'bar',
  data: {
    labels: LABELS,
    datasets: [
      {
        label: 'Tingkat Keyakinan',
        data: [0, 0, 0],
        backgroundColor: ['#ffb86c', '#ff79c6', '#8be9fd'],
        borderRadius: 5,
      },
    ],
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        beginAtZero: true,
        max: 1,
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#e0e0e0', font: { size: 14 } },
      },
      y: {
        grid: { display: false },
        ticks: { color: '#e0e0e0', font: { size: 14, weight: 'bold' } },
      },
    },
    plugins: { legend: { display: false } },
  },
});

// --- Fungsi Update UI ---
function updateUI(predictedClass, probabilities) {
  resultElement.textContent = predictedClass.toUpperCase();
  predictionChart.data.datasets[0].data = probabilities;
  predictionChart.update();
}

// --- Logika MediaPipe ---
function onResults(results) {
  // --- PERUBAHAN UTAMA DI SINI ---
  // Kita tidak lagi menggambar video di canvas. Canvas sekarang transparan
  // dan hanya digunakan untuk menggambar kerangka tangan.
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      // Gunakan warna default (hijau & merah)
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: '#00FF00',
        lineWidth: 5,
      });
      drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
    }
  }
  canvasCtx.restore();
}

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
});
hands.onResults(onResults);

// --- Loop Pemrosesan Video ---
async function processVideo() {
  if (!isCameraOn) return;
  // Karena video di-mirror oleh CSS, kita tidak perlu memproses versi mirror
  await hands.send({ image: videoElement });
  requestAnimationFrame(processVideo);
}

// --- Fungsi Prediksi Otomatis ke Backend ---
async function sendFrameToBackend() {
  if (!isCameraOn || isPredicting) return;
  isPredicting = true;
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = videoElement.videoWidth;
  tempCanvas.height = videoElement.videoHeight;
  const tempContext = tempCanvas.getContext('2d');

  // Kita tetap mengirim frame yang sudah di-mirror agar sesuai dengan apa yang dilihat pengguna
  tempContext.translate(tempCanvas.width, 0);
  tempContext.scale(-1, 1);
  tempContext.drawImage(
    videoElement,
    0,
    0,
    tempCanvas.width,
    tempCanvas.height
  );

  tempCanvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('file', blob, 'capture.jpg');
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.prediksi && data.probabilitas) {
        const labels_en = ['paper', 'rock', 'scissors'];
        const index = labels_en.indexOf(data.prediksi);
        const prediksi_id = LABELS[index];
        updateUI(prediksi_id, data.probabilitas);
      }
    } catch (error) {
      console.error('Error Prediksi:', error);
      resultElement.textContent = 'ERROR';
    } finally {
      isPredicting = false;
    }
  }, 'image/jpeg');
}

// --- Fungsi untuk Start/Stop Kamera ---
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    videoElement.srcObject = stream;
    videoElement.addEventListener('loadedmetadata', () => {
      isCameraOn = true;
      recordingIndicator.classList.remove('hidden');
      toggleCameraBtn.classList.add('text-red-500');
      resultElement.textContent = 'Menunggu...';
      processVideo();
    });
  } catch (err) {
    console.error('Gagal mengakses kamera: ', err);
    alert('Gagal mengakses kamera. Pastikan Anda memberikan izin.');
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  videoElement.srcObject = null;
  isCameraOn = false;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  recordingIndicator.classList.add('hidden');
  toggleCameraBtn.classList.remove('text-red-500');
  resultElement.textContent = 'Kamera Mati';
  updateUI('...', [0, 0, 0]);
}

// --- Fungsionalitas Tombol Kamera ---
toggleCameraBtn.addEventListener('click', () => {
  if (isCameraOn) {
    stopCamera();
  } else {
    startCamera();
  }
});

// --- Mulai Otomatis ---
document.addEventListener('DOMContentLoaded', () => {
  startCamera();
  setInterval(sendFrameToBackend, PREDICTION_INTERVAL);
});
