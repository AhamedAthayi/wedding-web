/* ================================================================
   Wedding AI Photo Finder — script.js
   Shared Utilities: Face API, Storage, DOM Helpers
   ================================================================ */

// ─── Face API Setup ────────────────────────────────────────────

// Models are loaded from the official face-api.js repository.
// On first load these may take 10–30 seconds depending on connection.
const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';

let _modelsLoaded  = false;
let _modelsLoading = false;
let _modelLoadPromise = null;

/**
 * Load face-api.js models (TinyFaceDetector + Landmarks + Recognition).
 * Safe to call multiple times — loads only once.
 * @param {Function} onStatus  - optional callback(string) for status updates
 */
async function loadFaceModels(onStatus) {
  if (_modelsLoaded) return;

  if (_modelsLoading) {
    await _modelLoadPromise;
    return;
  }

  _modelsLoading = true;
  _modelLoadPromise = (async () => {
    try {
      if (onStatus) onStatus('Loading face detection model… (1/3)');
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);

      if (onStatus) onStatus('Loading landmark model… (2/3)');
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

      if (onStatus) onStatus('Loading recognition model… (3/3)');
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

      _modelsLoaded  = true;
      _modelsLoading = false;
      if (onStatus) onStatus('Models ready ✓');
    } catch (err) {
      _modelsLoading = false;
      throw new Error('Failed to load face models. Check your internet connection.');
    }
  })();

  await _modelLoadPromise;
}

/**
 * Detect all faces in an image/video/canvas and return their descriptors.
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} input
 * @returns {Promise<number[][]>} Array of 128-float descriptors (as plain arrays)
 */
async function detectFaceDescriptors(input) {
  const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.45 });
  const detections = await faceapi
    .detectAllFaces(input, options)
    .withFaceLandmarks()
    .withFaceDescriptors();
  return detections.map(d => Array.from(d.descriptor));
}

/**
 * Detect a single face from a video element (used for live scanning).
 * Returns the best-confidence descriptor or null.
 * @param {HTMLVideoElement} video
 * @returns {Promise<number[]|null>}
 */
async function detectSingleFace(video) {
  const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
  const detection = await faceapi
    .detectSingleFace(video, options)
    .withFaceLandmarks()
    .withFaceDescriptor();
  return detection ? Array.from(detection.descriptor) : null;
}

/**
 * Euclidean distance between two descriptors.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
function descriptorDistance(a, b) {
  return faceapi.euclideanDistance(new Float32Array(a), new Float32Array(b));
}

/**
 * Check if a stored photo contains a face matching the query descriptor.
 * @param {Object}   photo           - photo object with .descriptors array
 * @param {number[]} queryDescriptor - descriptor from the user's face
 * @param {number}   threshold       - match threshold (lower = stricter; default 0.55)
 * @returns {boolean}
 */
function photoMatchesFace(photo, queryDescriptor, threshold = 0.55) {
  if (!photo.descriptors || photo.descriptors.length === 0) return false;
  return photo.descriptors.some(stored => descriptorDistance(stored, queryDescriptor) < threshold);
}

// ─── Image Helpers ─────────────────────────────────────────────

/**
 * Load an image from a URL / data URL into an HTMLImageElement.
 */
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload  = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/**
 * Read a File as a base-64 data URL.
 */
function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload  = e => resolve(e.target.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

/**
 * Resize a data URL to keep images manageable (max 1200px on longest side).
 */
function resizeDataUrl(dataUrl, maxPx = 1200) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxPx / Math.max(img.width, img.height));
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      c.getContext('2d').drawImage(img, 0, 0, w, h);
      resolve(c.toDataURL('image/jpeg', 0.88));
    };
    img.src = dataUrl;
  });
}

/**
 * Capture the current video frame as a JPEG data URL.
 */
function captureVideoFrame(video) {
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  return c.toDataURL('image/jpeg', 0.9);
}

/**
 * Generate a short unique ID.
 */
function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

// ─── LocalStorage Wrappers ─────────────────────────────────────

const KEYS = {
  PHOTOS:   'wai_photos',
  DETAILS:  'wai_details',
  FEEDBACK: 'wai_feedback',
  AUTH:     'wai_auth'
};

function getWeddingDetails() {
  try { return JSON.parse(localStorage.getItem(KEYS.DETAILS)) || {}; }
  catch { return {}; }
}

function saveWeddingDetails(details) {
  localStorage.setItem(KEYS.DETAILS, JSON.stringify(details));
}

function getPhotos() {
  try { return JSON.parse(localStorage.getItem(KEYS.PHOTOS)) || []; }
  catch { return []; }
}

/**
 * Append a photo to storage.
 * Returns true on success, false if storage quota was exceeded.
 */
function savePhoto(photo) {
  const photos = getPhotos();
  photos.push(photo);
  try {
    localStorage.setItem(KEYS.PHOTOS, JSON.stringify(photos));
    return true;
  } catch {
    showToast('Storage full! Try uploading smaller images or clear old photos.', 'error');
    return false;
  }
}

function deletePhoto(id) {
  const photos = getPhotos().filter(p => p.id !== id);
  localStorage.setItem(KEYS.PHOTOS, JSON.stringify(photos));
}

function clearAllPhotos() {
  localStorage.removeItem(KEYS.PHOTOS);
}

function getFeedbacks() {
  try { return JSON.parse(localStorage.getItem(KEYS.FEEDBACK)) || []; }
  catch { return []; }
}

function saveFeedback(fb) {
  const list = getFeedbacks();
  list.push({ ...fb, at: new Date().toISOString() });
  localStorage.setItem(KEYS.FEEDBACK, JSON.stringify(list));
}

// ─── Admin Auth ────────────────────────────────────────────────

function isAdminLoggedIn() {
  return localStorage.getItem(KEYS.AUTH) === '1';
}

function loginAdmin(user, pass) {
  // Change these credentials for production use
  if (user === 'admin' && pass === 'wedding2024') {
    localStorage.setItem(KEYS.AUTH, '1');
    return true;
  }
  return false;
}

function logoutAdmin() {
  localStorage.removeItem(KEYS.AUTH);
}

// ─── DOM Helpers ───────────────────────────────────────────────

const $ = id => document.getElementById(id);
const show = id => $( id)?.classList.remove('hidden');
const hide = id => $( id)?.classList.add('hidden');

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function setProgress(fillId, pct) {
  const el = $(fillId);
  if (el) el.style.width = Math.min(100, pct) + '%';
}

// ─── Toast Notifications ───────────────────────────────────────

let _toastContainer = null;

function getToastContainer() {
  if (!_toastContainer) {
    _toastContainer = document.createElement('div');
    _toastContainer.className = 'toast-container';
    document.body.appendChild(_toastContainer);
  }
  return _toastContainer;
}

/**
 * Show a temporary toast message.
 * @param {string} msg
 * @param {'success'|'error'|''} type
 * @param {number} duration  ms (default 3000)
 */
function showToast(msg, type = '', duration = 3000) {
  const container = getToastContainer();
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity .3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─── Webcam Utilities ──────────────────────────────────────────

let _webcamStream = null;

/**
 * Start webcam stream on a <video> element.
 * @param {HTMLVideoElement} video
 * @param {boolean} front  - prefer front camera
 */
async function startWebcam(video, front = true) {
  if (_webcamStream) stopWebcam(video);
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: front ? 'user' : 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  video.srcObject = stream;
  _webcamStream = stream;
  await video.play();
}

/**
 * Stop the active webcam stream.
 */
function stopWebcam(video) {
  if (_webcamStream) {
    _webcamStream.getTracks().forEach(t => t.stop());
    _webcamStream = null;
  }
  if (video) { video.srcObject = null; }
}

// ─── Download Helpers ──────────────────────────────────────────

/**
 * Trigger a browser download of a data URL.
 */
function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

/**
 * Download multiple photos as a ZIP file using JSZip.
 * @param {Object[]} photos  - array of {dataUrl, filename}
 * @param {string}   zipName
 */
async function downloadAsZip(photos, zipName = 'wedding-photos.zip') {
  if (typeof JSZip === 'undefined') {
    showToast('JSZip not loaded. Please refresh.', 'error');
    return;
  }
  const zip  = new JSZip();
  const folder = zip.folder('wedding-photos');

  photos.forEach((p, i) => {
    const base64 = p.dataUrl.split(',')[1];
    const ext    = p.dataUrl.includes('png') ? 'png' : 'jpg';
    const name   = p.filename || `photo-${i + 1}.${ext}`;
    folder.file(name, base64, { base64: true });
  });

  const blob = await zip.generateAsync({ type: 'blob' });
  const url  = URL.createObjectURL(blob);
  downloadDataUrl(url, zipName);
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  showToast(`Downloaded ${photos.length} photos as ZIP 📦`, 'success');
}

// ─── WhatsApp Share ────────────────────────────────────────────

function shareOnWhatsApp(text) {
  const encoded = encodeURIComponent(text);
  window.open(`https://wa.me/?text=${encoded}`, '_blank');
}
