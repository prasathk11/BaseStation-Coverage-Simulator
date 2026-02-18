# Base Station Coverage Simulator

A real-time visualization tool for simulating signal strength between base stations and mobile devices.

## 🚀 Features

* 🗼 Multiple base station simulation

* 📱 Dynamic mobile device movement with realistic physics

* 📊 Real-time signal strength visualization

* 🎨 Interactive controls and customization

* 🌐 Coverage area visualization

* ⚡ Adjustable simulation speed

* 🎯 Clean, modular architecture

## 📋 Requirements

* Python 3.7 or higher

* tkinter (usually comes with Python)

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/prasathk11/BaseStation-Coverage-Simulator.git
cd BaseStation-Coverage-Simulator
```

1. No additional dependencies required! The project uses only Python standard library.

## 🎮 Usage

Run the simulator:

```bash
python main.py
```

### Controls

* **⚡ Speed Slider**: Adjust the movement speed of mobile devices (0.1x - 3.0x)

* **📱 Mobiles Slider**: Change the number of mobile devices (1-10)

* **⏸ Pause/Resume Button**: Pause or resume the simulation

* **Coverage Circles**: Toggle visibility of base station coverage areas

* **Grid**: Toggle grid overlay for better spatial reference

## 📁 Project Structure

```
BaseStation-Coverage-Simulator/
├── main.py              # Entry point
├── simulator.py         # Main simulator class
├── mobile.py            # Mobile device model
├── renderer.py          # Canvas rendering functions
├── controls.py          # UI control panel
├── utils.py             # Utility functions (calculations)
├── config.py            # Configuration settings
├── README.md            # This file
├── LICENSE              # MIT License
└── .gitignore          # Git ignore file
```

## 🏗️ Architecture

### Modular Design

The project follows a clean, modular architecture:

* **`config.py`**: Centralized configuration (colors, settings, constants)

* **`utils.py`**: Pure utility functions (distance, signal strength calculations)

* **`mobile.py`**: Mobile device class with physics simulation

* **`renderer.py`**: Rendering logic separated from business logic

* **`controls.py`**: UI controls encapsulated in a class

* **`simulator.py`**: Main orchestrator that ties everything together

* **`main.py`**: Simple entry point


---

Made with ❤️ by Prasath K