# 👁️ Distributed Vision-Control System (Phase 1)

**Team ID:** `y3c_ajhm`
**Course:** Distributed Systems / Computer Vision
**Phase:** 1 (Open-Loop Actuation & Simulation)

> **Based on Previous Work:** This project builds upon the Face Locking behavioral analysis system. You can view the original foundation here:
> 🔗 [FaceLocking Repository](https://github.com/DannyMikeGanzaRwabuhama/FaceLocking.git)

---

## 1. 📝 Project Overview

This project implements a **distributed face-tracking system** where a PC-based vision engine detects a face, publishes movement commands to a central MQTT broker, and an ESP8266 edge controller moves a servo motor in response. A real-time web dashboard visualizes the system state via WebSocket.

The system is designed to be **loosely coupled**:

* 🧠 **Vision Node:** Produces data (Publisher)
* 🤖 **Edge Node:** Consumes commands (Subscriber)
* 📺 **Dashboard:** Visualizes state (Passive Subscriber)

---

## 2. 🏗️ System Architecture

The system follows a strict **Publisher-Subscriber** model using a central VPS broker.

### 🧩 Components

1. **Vision Node (PC) 💻**
* **Function:** Captures video from the webcam.
* **Tech Stack:** Uses **ArcFace** for face recognition and **MediaPipe** for tracking.
* **Logic:** Calculates positional error (Left/Right/Center).
* **Output:** **Publishes** JSON commands to the MQTT broker.
* *Constraint:* No direct connection to ESP8266 or Browser.


2. **Edge Node (ESP8266) 📡**
* **Function:** Connects to WiFi and the MQTT Broker.
* **Input:** **Subscribes** to movement topics.
* **Action:** Parses JSON payloads and drives the Servo Motor.
* *Constraint:* No HTTP/WebSocket servers allowed.


3. **Web Dashboard 📊**
* **Function:** Connects to the **WebSocket API Service (Port 9002)**.
* **Display:** Visualizes real-time servo commands and tracking confidence.
* *Constraint:* Does not connect to MQTT (Port 1883) directly.



---

## 3. 📡 MQTT Configuration

To ensure isolation on the shared broker, we use a unique team namespace.

* **Broker IP:** `157.173.101.159`
* **Port:** `1883`
* **Base Topic:** `vision/y3c_ajhm/`

### Published Topics

| Topic | Publisher | Payload Description |
| --- | --- | --- |
| `vision/y3c_ajhm/movement` | Vision Node | JSON command for servo control. |

**Payload Format:**

```json
{
  "status": "MOVE_LEFT",   // Enum: MOVE_LEFT, MOVE_RIGHT, CENTERED, NO_FACE
  "confidence": 0.98,      // ArcFace similarity score (0.0 - 1.0)
  "timestamp": 1730000000  // Unix Epoch Time
}

```

---

## 4. 📂 Repository Structure

```text
FaceLockingServo/
├── dashboard/
│   └── index.html             # 📊 Real-time Web Visualization
├── edge_node/
│   └── firmware.ino           # 🤖 ESP8266 Arduino Firmware
├── vision_node/
│   ├── main.py                # 🧠 Main Vision Engine Entry Point
│   ├── haar_5pt.py            # 📏 Face Alignment Helper
│   └── models/
│       └── embedder_arcface.onnx  # 🎭 Face Recognition Model
├── requirements.txt           # 📦 Python Dependencies
└── README.md                  # 📖 System Documentation

```

---

## 5. 🚀 Setup & Execution Instructions

### Prerequisites

* 🐍 Python 3.9+
* ⚡ Arduino IDE (for ESP8266)
* 📷 Webcam

### Step 1: Vision Node (PC) 🧠

1. **Install dependencies:**
```bash
pip install -r requirements.txt

```


2. **Run the vision engine:**
```bash
python vision_node/main.py

```


3. **Verify:** The terminal will show `📡 Connected to MQTT` and start printing `SENT: MOVE_...` logs.

### Step 2: Edge Node (ESP8266) 🤖

1. Open `edge_node/firmware.ino` in Arduino IDE.
2. Install the **PubSubClient** library via Library Manager.
3. Update the `ssid` and `password` variables at the top of the file.
4. Flash the code to your NodeMCU/ESP8266.
5. **Verify:** Open Serial Monitor (115200 baud) to see `✅ WiFi connected` and `Message received...`.

### Step 3: Web Dashboard 📺

1. Navigate to the `dashboard/` folder.
2. Open `index.html` in a modern web browser (Chrome/Edge/Firefox).
* **Important:** If using a strict browser, open as a local file (`file:///...`) to avoid Mixed Content errors with the non-SSL WebSocket.


3. **Verify:** The "Status" badge should turn Green/Connected, and arrows should update in real-time.

---

## 6. ⚠️ Known Issues & Network Limitations

During the evaluation phase, the following infrastructure constraints were documented:

1. **Dashboard WebSocket (Port 9002):**
The assignment requires connecting the dashboard to Port 9002 (`ws://157.173.101.159:9002`). During our testing, this port appeared unreachable/blocked from our network environment, preventing the dashboard from receiving data despite the code being correct.
2. **ESP8266 DHCP Failure:**
The ESP8266 hardware struggled to obtain an IP address (DHCP) on the campus "RCA" network, likely due to 5GHz/2.4GHz band steering issues or IP exhaustion.