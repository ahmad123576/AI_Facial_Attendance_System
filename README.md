# AI Face Attendance System

A Python-based face recognition attendance system using OpenCV and Tkinter.

## Features

- ✅ Enroll students with face capture (20 images per student)
- ✅ Train face recognition model using LBPH algorithm
- ✅ Mark attendance using live camera feed
- ✅ View attendance with present/absent lists
- ✅ Export attendance to Excel files
- ✅ Manage subjects and settings

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <your-repo-url>
   cd face-attendance
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv env
   ```

3. **Activate virtual environment**
   - Windows: `env\Scripts\activate`
   - Linux/Mac: `source env/bin/activate`

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**
   ```bash
   python main.py
   ```

2. **First-time setup:**
   - The `data/` folder will be created automatically when you run the app
   - Click **"Enroll New Student"** to add students (captures 20 face images)
   - Click **"Train Model"** after enrolling students
   - Click **"Mark Attendance"** to take attendance

3. **Workflow:**
   - Enroll → Train → Mark Attendance → View/Export

## Project Structure

```
face-attendance/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── data/                  # Created automatically (not in git)
    ├── TrainingImage/     # Student face images
    ├── trainer.yml        # Trained model
    ├── config.json        # Settings
    ├── subjects.json      # Subject list
    └── *.xlsx            # Attendance Excel files
```

## Requirements

- Python 3.11.9
- OpenCV (opencv-contrib-python)
- NumPy
- Pandas
- Pillow (PIL)
- openpyxl

## Notes

- The `data/` folder is automatically created when you first run the app
- You need at least 2 enrolled students for reliable recognition
- Default recognition threshold is 40 (lower = stricter, higher = more lenient)
- Adjust settings via **⚙️ Settings** button

