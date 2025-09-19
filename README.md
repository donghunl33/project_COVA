# COVA: Vehicle Authentication and Control System

**A system that determines driving eligibility through RFID-based user authentication and sensor diagnostics, enabling safe vehicle control.**

---

## 📽️ Demo Video  
[![demo](https://github.com/addinedu-ros-9th/iot-repo-1/raw/main/images/car.png)](https://youtu.be/cT2-3-PcFhQ)

---

## 🔑 Key Features (Summary)

- **RFID Authentication**: Only registered cards can control the vehicle  
- **Alcohol Detection**: Blocks ignition if alcohol is detected  
- **Vehicle Control**: Forward / Reverse / Left / Right / Stop  
- **Sensor Data Logging**: Save sensor values (shock, temperature, etc.) in DB  
- **GUI-based Status Display**: Real-time visualization of vehicle state  
- **RFID Registration/Lookup**: Register and manage user RFID cards  

---

## 📑 Table of Contents

1. [Overview](#1-overview)  
2. [Key Features](#2-key-features)  
3. [Team Information](#3-team-information)  
4. [Development Environment](#4-development-environment)  
5. [System Design](#5-system-design)  
   - [User Requirements](#51-user-requirements)  
   - [System Requirements](#52-system-requirements)  
   - [System Architecture](#53-system-architecture)  
   - [Scenario](#54-scenario)  
   - [GUI](#55-gui)  
6. [Database Design](#6-database-design)  
   - [ER Diagram](#61-er-diagram)  
7. [Interface Specification](#7-interface-specification)  
8. [Test Cases](#8-test-cases)  
9. [Problems and Solutions](#9-problems-and-solutions)  
10. [Limitations](#10-limitations)  
11. [Conclusion and Future Work](#11-conclusion-and-future-work)

---

## 1. Overview

COVA controls and records safe vehicle usage based on **driver authentication and condition monitoring**.  
It consists of Arduino-based sensors and RFID authentication devices, collectively determining user eligibility and monitoring driving behavior.  

---

## 2. Key Features

- RFID-based user authentication  
- Alcohol detection (MQ2)  
- Temperature / Light / Shock sensor data collection  
- Motor-based vehicle movement control  
- GUI for real-time feedback  
- Database integration for sensor logging  

---

## 3. Team Information

| Name     | Role    | Responsibilities |
|----------|---------|------------------|
| **Younghun Yoo** | Team Lead | RFID authentication, Alcohol detection, Software, Test cases, README |
| **Jineon Kim**   | Member   | Server development, DB setup, Integration & testing |
| **Taeho Kim**    | Member   | RFID registration + GUI, DB integration, Presentation, Scenario/ERD/GUI docs |
| **Dongyeon Lee** | Member   | Headlight control, Reverse alert, Temperature detection, Presentation, Interface spec |
| **Donghun Lee**  | Member   | Motor control + GUI, Shock detection, Software integration, System Architecture, Requirements |

---

## 4. Development Environment

### 4.1 Hardware Setup
| Item   | Components |
|--------|------------|
| PC     | 3 units |
| Boards | Arduino Uno, Arduino Mega |
| Sensors | Gas, Light, Temperature, Ultrasonic, RFID |
| Motor  | Motors, Motor driver |
| Others | LED, Buzzer |

### 4.2 Software Setup
| Item | Details |
|------|---------|
| Programming | C++ (Arduino), Python (Server/Validation) |
| Database    | MySQL |
| Tools       | Arduino IDE, VSCode, Git, GitHub |

### 4.3 Tools & Tech Stack
| Category | Technology |
|----------|------------|
| OS & IDE | ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=flat&logo=ubuntu&logoColor=white) ![VSCode](https://img.shields.io/badge/VSCode-007ACC?style=flat&logo=visual-studio-code&logoColor=white) |
| Language | ![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=cplusplus&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| Backend  | ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) |
| Database | ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white) |
| Version Control | ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) |
| Collaboration | ![Confluence](https://img.shields.io/badge/Confluence-172B4D?style=flat&logo=confluence&logoColor=white) ![Jira](https://img.shields.io/badge/Jira-0052CC?style=flat&logo=jira&logoColor=white) ![Slack](https://img.shields.io/badge/Slack-4A154B?style=flat&logo=slack&logoColor=white) |

---

## 5. System Design

### 5.1 User Requirements
(Translated list of UR_1 ~ UR_7)

### 5.2 System Requirements
(Translated list of SR_1 ~ SR_9)

### 5.3 System Architecture
- **Hardware**  
![HW_architecture](https://github.com/addinedu-ros-9th/iot-repo-1/blob/main/images/HW_architecture.png)

- **Software**  
![SW_architecture](https://github.com/addinedu-ros-9th/iot-repo-1/blob/main/images/SW_architecture.png)

### 5.4 Scenarios
- RFID Register & Browse  
- Vehicle Authentication  
- Shock & Temperature Management  
- Vehicle Control  
- Headlight Control  
- Reverse Management  
(images included in original)

### 5.5 GUI
- **COVA Admin**: Admin default, RFID check, Registration  
- **COVA Jr Control**: Default, Door open, Engine start, Headlight on, Reverse  
(images included in original)

---

## 6. Database Design

### 6.1 ER Diagrams
- **Admin DB**  
![Admin DB](https://github.com/addinedu-ros-9th/iot-repo-1/blob/main/images/Admin%20DB.png)  
- **Vehicle DB**  
![Vehicle DB](https://github.com/addinedu-ros-9th/iot-repo-1/blob/main/images/Vehicle%20DB.png)  

---

## 7. Interface Specification
(Translated tables for command list, verification, logging, ultrasonic, sensors, motor, server communication, etc. — same format as original)

---

## 8. Test Cases
(Translated test case table with Pass/Fail results)

---

## 9. Problems and Solutions
- **Issue:** ESP overloaded with tasks, hardware limits  
  **Solution:** Removed ESP, replaced with PC-to-PC **TCP communication**  
- **Issue:** HTTP caused delays  
  **Solution:** Used **serial communication** (Arduino↔PC) and **TCP** (PC↔PC) for real-time performance  
- **Issue:** Arduino Uno couldn’t handle all sensors + comms  
  **Solution:** Split roles: **Arduino Mega = sensors, Arduino Uno = motors**  

---

## 10. Limitations
1. More precise alcohol detection & anti-tampering needed  
2. Additional GPS, speed, and acceleration sensors for enriched data  
3. Security for servers managing driver personal data  
4. More driving habit data needed for analysis  
5. Improve motor response to ultrasonic data in real-time  
6. Replace low-quality sensors with higher-grade sensors  

---

## 11. Conclusion
- Successfully developed **smart authentication system** using RFID smart keys  
- Achieved **real-time collection & storage of multi-sensor data**  
- Improved system stability & scalability via distributed hardware and optimized communication  
- Provides a foundation for **enhanced driver safety and future data-driven services**  
