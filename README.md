# Medical Imaging Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.2.0-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An AI-powered medical imaging diagnostic system leveraging deep learning for multi-modal medical image analysis with intelligent chatbot assistance.**

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training Models](#training-models)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

The **Medical Imaging Diagnostic System** is an advanced B.Tech final year project that combines state-of-the-art deep learning techniques with modern web technologies to provide accurate medical image analysis across multiple imaging modalities. The system offers:

- **Multi-modal Image Classification**: Supports retinal scans, dermoscopy images, and X-rays
- **AI-Powered Diagnosis**: Utilizes transfer learning with EfficientNet-B0 architecture
- **Medical Report Analysis**: Extracts and interprets PDF medical reports
- **Interactive AI Chatbot**: Provides medical guidance powered by Google Gemini API
- **Real-time Predictions**: Instant diagnosis with confidence scores and probability distributions

## Features

### Medical Imaging Analysis

| Modality | Dataset | Disease Detection | Classes |
|----------|---------|-------------------|---------|
| **Retinal Scans** | APTOS | Diabetic Retinopathy | 5 severity levels |
| **Dermoscopy** | HAM10000 | Skin Lesions | 7 lesion types |
| **X-Ray** | MURA | Bone Fractures | Binary (Normal/Abnormal) |

### AI-Powered Features

- **Deep Learning Models**: Transfer learning with pre-trained EfficientNet-B0
- **High Accuracy**: 75-90% validation accuracy across datasets
- **Confidence Scoring**: Probability distribution for all possible diagnoses
- **PDF Text Extraction**: Automated medical report parsing
- **Conversational AI**: Context-aware chatbot using Google Gemini
- **Responsive UI**: Modern React-based interface with real-time updates

### User Experience

- **Drag & Drop Upload**: Intuitive file upload interface
- **Real-time Analysis**: Instant predictions with visual feedback
- **Interactive Results**: Detailed probability breakdowns with color-coded confidence
- **Chat Interface**: Ask questions about diagnosis and treatment
- **Multi-device Support**: Responsive design for desktop and mobile

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  Image     │  │    PDF     │  │   Chat Interface     │  │
│  │  Uploader  │  │  Uploader  │  │   (Gemini AI)        │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    REST API (FastAPI)                        │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  /predict  │  │ /pdf/extract│  │      /chat           │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Services                           │
│  ┌────────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Deep Learning  │  │ PDF Parser  │  │ Gemini Chatbot  │  │
│  │ (PyTorch)      │  │ (PyPDF2)    │  │ (Google AI)     │  │
│  └────────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend
- **Framework**: FastAPI 0.104.1
- **Deep Learning**: PyTorch 2.1.0, TorchVision 0.16.0
- **ML Models**: EfficientNet-B0 (Transfer Learning)
- **Computer Vision**: OpenCV 4.8.1
- **PDF Processing**: PyPDF2 3.0.1
- **AI Integration**: Google Generative AI 0.3.1

### Frontend
- **Framework**: React 18.2.0
- **Build Tool**: Vite 5.0.8
- **Styling**: Tailwind CSS 3.3.6
- **HTTP Client**: Axios 1.6.2
- **File Upload**: React Dropzone 14.2.3

### Data & Training
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Augmentation**: TorchVision Transforms

---

## Dataset Information

### 1. APTOS (Diabetic Retinopathy)
- **Source**: Kaggle APTOS 2019 Blindness Detection
- **Images**: ~3,000 retinal fundus photographs
- **Classes**: 5 (No DR, Mild, Moderate, Severe, Proliferative DR)
- **Format**: PNG images with CSV labels

### 2. HAM10000 (Skin Lesions)
- **Source**: Harvard Dataverse
- **Images**: 10,015 dermoscopic images
- **Classes**: 7 (akiec, bcc, bkl, df, mel, nv, vasc)
- **Format**: JPG images with metadata CSV

### 3. MURA (Musculoskeletal Radiographs)
- **Source**: Stanford ML Group
- **Images**: 40,000+ X-ray images
- **Body Parts**: Shoulder, Elbow, Finger, Forearm, Hand, Humerus, Wrist
- **Classes**: Binary (Normal/Abnormal)
- **Format**: PNG images with study-level labels

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 16+ and npm
- GPU recommended (CUDA-compatible) for training
- 10GB+ free disk space (excluding datasets)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/medical-imaging-diagnostic-system.git
cd medical-imaging-diagnostic-system
```

### Step 2: Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### Step 4: Download Datasets

Download the following datasets and place them in the `Datasets/` directory:

1. **APTOS**: Place in `Datasets/APTOS_dataset/`
2. **HAM10000**: Place in `Datasets/HAM10000_dataset/`
3. **MURA**: Place in `Datasets/MURA_dataset/`

---

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
# Google Gemini API Configuration
GOOGLE_API_KEY=your-api-key-here
```

**Get your free API key**: https://makersuite.google.com/app/apikey

> **Note**: The chatbot works without an API key using intelligent fallback responses.

---

## Training Models

### Train Individual Models

```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Train APTOS model (Retinal scans)
python backend/train.py --dataset aptos --epochs 20

# Train HAM10000 model (Skin lesions)
python backend/train.py --dataset ham10000 --epochs 20

# Train MURA model (X-rays)
python backend/train.py --dataset mura --epochs 20
```

### Training Options

```bash
python backend/train.py --dataset <dataset> --epochs <num> --model <model_name> --resume
```

**Arguments:**
- `--dataset`: Dataset to train on (`aptos`, `ham10000`, `mura`)
- `--epochs`: Number of training epochs (default: 20)
- `--model`: Model architecture (`efficientnet`, `resnet50`, `densenet121`)
- `--resume`: Resume from checkpoint

### Expected Training Time

| Hardware | Time per Model |
|----------|----------------|
| GPU (NVIDIA RTX 3060+) | 2-4 hours |
| GPU (NVIDIA GTX 1060) | 4-6 hours |
| CPU (Modern i7/i9) | 8-12 hours |

**Trained models** are saved in `backend/saved_models/`

---

## Running the Application

### Method 1: Using Batch Scripts (Windows)

```bash
# Terminal 1: Start Backend
start_backend.bat

# Terminal 2: Start Frontend
start_frontend.bat
```

### Method 2: Manual Start

**Terminal 1 - Backend Server:**
```bash
cd "path/to/project"
.\venv\Scripts\Activate.ps1
python backend/main.py
```
Backend runs on: http://localhost:8000

**Terminal 2 - Frontend Server:**
```bash
cd "path/to/project/frontend"
npm run dev
```
Frontend runs on: http://localhost:3000

### Access the Application

Open your browser and navigate to: **http://localhost:3000**

---



## Project Structure

```
medical-imaging-diagnostic-system/
├── backend/
│   ├── models/
│   │   └── classifier.py          # Model architectures
│   ├── utils/
│   │   ├── data_loader.py         # Dataset loaders & transforms
│   │   ├── predictor.py           # Inference engine
│   │   ├── pdf_extractor.py       # PDF text extraction
│   │   └── chatbot.py             # AI chatbot logic
│   ├── saved_models/              # Trained model checkpoints
│   ├── uploads/                   # Temporary uploaded files
│   ├── config.py                  # Configuration constants
│   ├── train.py                   # Training script
│   └── main.py                    # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── ResultsDisplay.jsx
│   │   │   ├── PDFUploader.jsx
│   │   │   └── ChatInterface.jsx
│   │   ├── App.jsx                # Main application
│   │   ├── api.js                 # API client
│   │   └── main.jsx               # Entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── Datasets/                      # Medical imaging datasets
│   ├── APTOS_dataset/
│   ├── HAM10000_dataset/
│   └── MURA_dataset/
├── .env                           # Environment variables (not in git)
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── setup.bat                      # Automated setup script
├── train_all_models.bat          # Train all models
└── README.md                      # This file
```

---

## Model Performance

### Validation Accuracy

| Model | Dataset | Accuracy | Parameters |
|-------|---------|----------|------------|
| EfficientNet-B0 | APTOS | 82.3% | 5.3M |
| EfficientNet-B0 | HAM10000 | 87.5% | 5.3M |
| EfficientNet-B0 | MURA | 79.8% | 5.3M |

### Training Metrics

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Image Size**: 224x224
- **Data Augmentation**: RandomFlip, RandomRotation, ColorJitter

---

## Screenshots

### Main Dashboard
![Dashboard](docs/screenshots/dashboard.png)

### Image Analysis
![Analysis](docs/screenshots/analysis.png)

### AI Chatbot
![Chatbot](docs/screenshots/chatbot.png)

---


## Acknowledgments

### Datasets
- **APTOS 2019 Blindness Detection** - Asia Pacific Tele-Ophthalmology Society
- **HAM10000** - Harvard Dataverse
- **MURA** - Stanford ML Group

### Technologies
- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Web Framework
- [React](https://reactjs.org/) - UI Library
- [Google Gemini](https://ai.google.dev/) - AI Language Model

### Inspiration
This project was developed as a B.Tech Final Year Project to demonstrate the practical application of AI/ML in healthcare diagnostics.

---


## Medical Disclaimer

**IMPORTANT**: This is an educational AI project designed for academic purposes only. The system should **NOT** be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment.

---

<div align="center">

**Made with ❤️ for Better Healthcare**

⭐ Star this repository if you found it helpful!

</div>
