# Dental Caries AI: Medical Safety Gate & System Telemetry Dashboard

A clinical demonstration system featuring a **7-Step Medical Input Validation Safety Gate** and a real-time **Node.js Process Telemetry & AI Model Dashboard**.

---

## 🎯 Key Capabilities

1. **Medical AI Input Validation Gate**:
   - Accepts **ONLY** Panoramic Dental Radiographs (OPG) in PNG/JPEG format.
   - Enforces strict rejection of Out-Of-Distribution (OOD) images (selfies, screenshots, intraoral photos, bitewings, documents).
   - Validates **Panoramic Dentition ROI**, **Detection Confidence (≥ 0.80)**, **Coverage Ratio (≥ 0.20)**, and **Mask Segmentation Ratio (≥ 0.15)**.
   - **Safety Enforcement**: Strictly prevents the Caries Surface Classification model from running if input validation fails.
2. **Real-Time Heartbeat Monitoring**:
   - 1-second pulse tracker with visual status indicator (**Emerald Green** for < 5s, **Red** for stale / disconnected).
3. **V8 Memory & CPU Runtime Gauges**:
   - Real-time `process.memoryUsage()` (Heap Used, Heap Total, RSS) and `process.cpuUsage()` (User and System time).
4. **Interactive AI Model Query Endpoint**:
   - `POST /chat` endpoint with diagnostic reasoning capabilities.
5. **Ingress Audit Stream**:
   - In-memory 100-entry ring buffer exposed at `GET /logs` with route, method, latency, and status code metrics.
6. **5-Second Auto Refresh**:
   - Automated telemetry synchronization with manual pause/resume and instant trigger controls.

---

## 📂 Project Structure

```
d:\Mahidol University\SP\caries_surface_classification\app/
├── package.json                   # Root workspace launcher
├── README.md                      # Documentation & run instructions
├── server/                        # Node.js + Express Backend
│   ├── package.json
│   ├── index.js                   # Express API & route handlers
│   ├── services/
│   │   ├── validationService.js   # 7-Step Medical Input Validation Gate
│   │   ├── cariesService.js       # Caries Surface Classifier (Safety-locked)
│   │   ├── metricsService.js      # Memory, CPU, and network telemetry
│   │   ├── loggerService.js       # 100-log ring buffer logger
│   │   └── modelService.js        # AI Model chat provider
│   └── test/
│       ├── test_validation.js     # Verification script for OPG, Selfie, Screenshot, Intraoral
│       └── smoke_test.js          # Verification script for API endpoints
└── client/                        # React + TypeScript + Tailwind CSS Frontend
    ├── package.json
    ├── vite.config.ts             # Vite dev server with backend API proxy
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx                # Main dashboard view
        ├── types/index.ts         # TypeScript data contracts
        └── components/            # Modular dashboard widgets
```

---

## 🚀 How to Run

### Step 1: Install Dependencies

```powershell
cd "d:\Mahidol University\SP\caries_surface_classification\app"

# Install root dependencies
npm install

# Install server dependencies
cd server
npm install

# Install client dependencies
cd ../client
npm install
```

### Step 2: Start Backend Server

```powershell
cd "d:\Mahidol University\SP\caries_surface_classification\app\server"
node index.js
```
The server will start on `http://localhost:3001`.

### Step 3: Start Frontend Dashboard

```powershell
cd "d:\Mahidol University\SP\caries_surface_classification\app\client"
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 🧪 Automated Verification Scripts

### 1. Run Medical Validation Safety Gate Test:
```powershell
cd "d:\Mahidol University\SP\caries_surface_classification\app\server"
node test/test_validation.js
```

### 2. Run API Smoke Test:
```powershell
cd "d:\Mahidol University\SP\caries_surface_classification\app\server"
node test/smoke_test.js
```
