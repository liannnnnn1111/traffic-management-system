# 🚦 Lian AI Traffic Monitor

A real-time traffic detection system built with Python, OpenCV, Ultralytics YOLOv8, and Tkinter. This tool detects and classifies vehicles from a live webcam or CCTV feed, tracks their speed, and flags overspeeding vehicles with snapshot logging and video recording.

---

## ✨ Features

- 🚗 Detects vehicles (car, motorcycle, bicycle, truck, e-bike)
- 📸 Captures snapshots of overspeeding vehicles
- 🎥 Records traffic videos automatically
- 📊 Shows live count of detected vehicles in GUI
- 📁 Saves CSV logs of detection history
- 🧠 Uses YOLOv8 model for high accuracy
- 👀 Supports both Webcam and CCTV (IP camera) input

---

## 🖥️ GUI Interface

- Built with `Tkinter`
- User-friendly interface
- Options for:
  - Start detection
  - Stop detection (auto-save)
  - Save snapshot manually
  - Reset vehicle count
  - Switch between webcam and CCTV

---

## ⚙️ Requirements

- Python 3.10+
- `ultralytics` (YOLOv8)
- OpenCV
- Pandas
- Tkinter (default in most Python installs)

Install dependencies via pip:

```bash
pip install ultralytics opencv-python pandas
