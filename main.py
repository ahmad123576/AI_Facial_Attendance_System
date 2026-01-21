import tkinter as tk
from tkinter import messagebox, ttk, simpledialog, filedialog
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json

class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Attendance - Python Edition")
        self.root.geometry("800x600")
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.training_dir = os.path.join(self.project_dir, "data", "TrainingImage")
        self.config_dir = os.path.join(self.project_dir, "data")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.subjects_file = os.path.join(self.config_dir, "subjects.json")
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Standard face image size for training (improves consistency)
        self.face_size = (200, 200)
        
        # Default settings
        self.camera_settings = {
            'width': 640,
            'height': 480,
            'brightness': 0,
            'contrast': 0
        }
        # LBPH confidence is a distance: LOWER = better match.
        # Using a stricter default (40) makes recognition more reliable and
        # helps avoid marking unknown faces as enrolled students.
        self.recognition_threshold = 40  # Default threshold for attendance
        
        # Load settings
        self.load_settings()
        
        # Recognizer with optimized parameters
        # radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=80
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=80.0
        )
        self.trained = False
        self.label_dict = {}  # Store label mapping: name -> label_id
        self.id_to_name = {}  # Reverse mapping: label_id -> name
        
        # Subject management
        self.subjects = []
        self.load_subjects()
        
        # Try to load existing trained model
        self.load_trained_model()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="AI Face Attendance System", font=("Arial", 20, "bold"), bg="#667eea", fg="white")
        header.pack(pady=20, fill="x")
        
        # Buttons Frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="Enroll New Student", command=self.enroll_student, bg="#38ef7d", fg="white", font=("Arial", 12), width=20, height=2).pack(pady=10)
        tk.Button(btn_frame, text="Train Model", command=self.train_model, bg="#667eea", fg="white", font=("Arial", 12), width=20, height=2).pack(pady=10)
        tk.Button(btn_frame, text="Mark Attendance", command=self.mark_attendance, bg="#ff6b6b", fg="white", font=("Arial", 12), width=20, height=2).pack(pady=10)
        tk.Button(btn_frame, text="Delete Student", command=self.delete_student, bg="#e74c3c", fg="white", font=("Arial", 12), width=20, height=2).pack(pady=10)
        tk.Button(btn_frame, text="View Attendance", command=self.view_attendance, bg="#4ecdc4", fg="white", font=("Arial", 12), width=20, height=2).pack(pady=10)
        tk.Button(btn_frame, text="View/Export Excel", command=self.export_excel, bg="#95a5a6", fg="white", font=("Arial", 12), width=20, height=2).pack(pady=10)
        
        # Settings and Management Frame
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(pady=10)
        
        tk.Button(settings_frame, text="⚙️ Settings", command=self.open_settings, bg="#9b59b6", fg="white", font=("Arial", 10), width=18, height=1).pack(side=tk.LEFT, padx=5)
        tk.Button(settings_frame, text="📚 Manage Subjects", command=self.manage_subjects, bg="#3498db", fg="white", font=("Arial", 10), width=18, height=1).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 10), fg="green")
        self.status_label.pack(pady=10)
    
    def load_settings(self):
        """Load settings from config file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.camera_settings = config.get('camera_settings', self.camera_settings)
                    self.recognition_threshold = config.get('recognition_threshold', 40)
            except Exception as e:
                print(f"Error loading settings: {e}")
    
    def save_settings(self):
        """Save settings to config file"""
        try:
            config = {
                'camera_settings': self.camera_settings,
                'recognition_threshold': self.recognition_threshold
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def load_subjects(self):
        """Load subjects from file"""
        if os.path.exists(self.subjects_file):
            try:
                with open(self.subjects_file, 'r') as f:
                    self.subjects = json.load(f)
            except Exception as e:
                print(f"Error loading subjects: {e}")
                self.subjects = []
        else:
            self.subjects = []
    
    def save_subjects(self):
        """Save subjects to file"""
        try:
            with open(self.subjects_file, 'w') as f:
                json.dump(self.subjects, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save subjects: {e}")
    
    # Note: previous 'class roster' loading/saving helpers were removed to keep
    # the app focused on a single, simple student list based on enrolled images.
    
    def get_existing_student_ids(self):
        """Get all existing student IDs from training images"""
        existing_ids = set()
        if os.path.exists(self.training_dir):
            for filename in os.listdir(self.training_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    parts = filename.split('_')
                    if len(parts) > 0:
                        existing_ids.add(parts[0])
        return existing_ids
    
    def apply_camera_settings(self, cap):
        """Apply camera settings to the capture object"""
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_settings['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_settings['height'])
            cap.set(cv2.CAP_PROP_BRIGHTNESS, self.camera_settings['brightness'])
            cap.set(cv2.CAP_PROP_CONTRAST, self.camera_settings['contrast'])
        except Exception as e:
            print(f"Warning: Could not apply all camera settings: {e}")
    
    def preprocess_face(self, face_image):
        """Preprocess face image for better recognition"""
        # Resize to standard size
        face_resized = cv2.resize(face_image, self.face_size)
        
        # Apply histogram equalization to improve contrast
        face_equalized = cv2.equalizeHist(face_resized)
        
        # Apply slight Gaussian blur to reduce noise
        face_blurred = cv2.GaussianBlur(face_equalized, (3, 3), 0)
        
        return face_blurred
    
    def enroll_student(self):
        enroll_win = tk.Toplevel(self.root)
        enroll_win.title("Enroll Student")
        enroll_win.geometry("400x500")
        
        tk.Label(enroll_win, text="Student ID:").pack(pady=10)
        self.student_id = tk.Entry(enroll_win)
        self.student_id.pack()
        
        tk.Label(enroll_win, text="Student Name:").pack(pady=10)
        self.student_name = tk.Entry(enroll_win)
        self.student_name.pack()
        
        tk.Button(enroll_win, text="Start Capture (20 Images)", command=lambda: self.capture_faces(enroll_win), bg="#38ef7d", fg="white").pack(pady=20)
        
        self.progress = ttk.Progressbar(enroll_win, length=300, mode='determinate')
        self.progress.pack(pady=10)
        
        self.status_text = tk.Label(enroll_win, text="", font=("Arial", 9))
        self.status_text.pack(pady=5)
    
    def capture_faces(self, win):
        name = self.student_name.get().strip()
        id_ = self.student_id.get().strip()
        
        if not name or not id_:
            messagebox.showerror("Error", "Enter ID and Name!")
            return
        
        # Check if student ID already exists
        existing_ids = self.get_existing_student_ids()
        if id_ in existing_ids:
            # Do NOT allow duplicate IDs to be enrolled again to avoid confusion.
            # User must choose a new ID if they want to add another student.
            messagebox.showerror(
                "Duplicate ID",
                f"Student ID '{id_}' is already enrolled.\n\n"
                "Please use a different ID for new students."
            )
            return
        
        # Sanitize name for filename (remove spaces, special chars)
        name_safe = name.replace(" ", "_").replace("/", "_")
        
        self.progress['maximum'] = 20
        self.progress['value'] = 0
        count = 0
        frame_count = 0
        last_capture_time = 0
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            win.destroy()
            return
        
        # Use better face detection parameters
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.status_text.config(text="Position face in front of camera. Move slowly...")
        win.update()
        
        while count < 20:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Improve face detection with better parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            current_time = time.time()
            
            # Only capture if single face detected and enough time passed (0.5 seconds)
            if len(faces) == 1 and (current_time - last_capture_time) > 0.5:
                x, y, w, h = faces[0]
                
                # Ensure face is large enough and centered
                if w > 100 and h > 100:
                    # Extract and preprocess face
                    face_roi = gray[y:y+h, x:x+w]
                    face_processed = self.preprocess_face(face_roi)
                    
                    # Save with zero-padded number for proper sorting
                    filename = os.path.join(self.training_dir, f"{id_}_{name_safe}_{count:03d}.jpg")
                    
                    # Ensure directory exists
                    os.makedirs(self.training_dir, exist_ok=True)
                    
                    # Save image and verify it was written
                    success = cv2.imwrite(filename, face_processed)
                    if success and os.path.exists(filename):
                        count += 1
                        self.progress['value'] = count
                        last_capture_time = current_time
                        self.status_text.config(text=f"Captured {count}/20 images - Saved")
                        win.update()
                    else:
                        self.status_text.config(text=f"Warning: Failed to save image {count+1}")
                        win.update()
                        print(f"Failed to save image: {filename}")
                    
                    # Flash green rectangle to show capture
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, f"Captured {count}/20", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Face too small
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                    cv2.putText(frame, "Move closer", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            elif len(faces) > 1:
                # Multiple faces detected
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Multiple faces detected!", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Single face but not ready to capture
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display capture progress on frame
            cv2.putText(frame, f"Progress: {count}/20", (10, frame.shape[0] - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'Capturing Faces - {name} (Turn Head Slowly)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Verify images were actually saved
        saved_files = []
        if os.path.exists(self.training_dir):
            saved_files = [f for f in os.listdir(self.training_dir) 
                          if f.startswith(f"{id_}_{name_safe}_") and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if count < 20:
            messagebox.showwarning("Warning", f"Only {count}/20 images were attempted to be captured!\n\nSaved files: {len(saved_files)}")
        else:
            if len(saved_files) >= count:
                messagebox.showinfo("Success", f"{count} images captured and saved for {name}!\n\nFiles saved: {len(saved_files)}\n\nYou can now train the model.")
            else:
                messagebox.showwarning("Warning", f"Attempted to capture {count} images, but only {len(saved_files)} were saved.\n\nPlease check the training directory.")
        
        # Reset trained status since new images were added
        self.trained = False
        win.destroy()
    
    def train_model(self):
        # Check if training directory exists
        if not os.path.exists(self.training_dir):
            messagebox.showerror("Error", f"Training directory does not exist!\n\nDirectory: {self.training_dir}\n\nPlease enroll students first.")
            self.status_label.config(text="Error: Directory not found", fg="red")
            return
        
        # Ensure directory is accessible
        try:
            all_files = os.listdir(self.training_dir)
        except PermissionError:
            messagebox.showerror("Error", f"Permission denied accessing training directory!\n\nDirectory: {self.training_dir}")
            self.status_label.config(text="Error: Permission denied", fg="red")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Cannot access training directory!\n\nError: {str(e)}\n\nDirectory: {self.training_dir}")
            self.status_label.config(text="Error: Cannot access directory", fg="red")
            return
        
        # Get all files and filter for image files only
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            error_msg = "No image files found!\n\n"
            error_msg += f"Directory: {self.training_dir}\n\n"
            if all_files:
                error_msg += f"Found {len(all_files)} file(s) but none are images:\n"
                error_msg += "\n".join([f"  - {f}" for f in all_files[:10]])
                if len(all_files) > 10:
                    error_msg += f"\n  ... and {len(all_files) - 10} more"
            else:
                error_msg += "Directory is empty.\n\nPlease enroll students first using the 'Enroll New Student' button."
            
            # Show where images should be saved
            error_msg += f"\n\nImages should be saved as: ID_NAME_NUMBER.jpg"
            
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text="Error: No images found", fg="red")
            return
        
        self.status_label.config(text="Training model...", fg="orange")
        self.root.update()
        
        faces = []
        labels = []
        # We will label people by **Student ID** (more reliable than name).
        # id_to_name: {label(int): "Display Name"}
        self.label_dict = {}  # kept for backward compatibility with older code paths
        self.id_to_name = {}
        
        # Sort filenames to ensure consistent label assignment (only image files)
        filenames = sorted(image_files)
        
        # First pass: build label mapping using Student ID from filename: ID_NAME_NUMBER.jpg
        for filename in filenames:
            parts = filename.split('_')
            if len(parts) >= 2:
                student_id_str = parts[0].strip()
                student_name = parts[1].replace("_", " ").strip()

                # Use numeric label if possible (OpenCV labels are int32)
                try:
                    label = int(student_id_str)
                except ValueError:
                    # If someone used non-numeric IDs, fall back to a stable hash-like mapping
                    # based on sorted unique IDs. (We rebuild deterministically each train.)
                    label = None

                # Temporarily store as string key; we'll finalize non-numeric IDs after this loop.
                if label is not None:
                    self.id_to_name[label] = student_name

        # Handle non-numeric IDs (rare): assign incremental labels starting after max numeric id
        non_numeric_ids = []
        for filename in filenames:
            parts = filename.split('_')
            if len(parts) >= 2:
                student_id_str = parts[0].strip()
                try:
                    int(student_id_str)
                except ValueError:
                    non_numeric_ids.append(student_id_str)

        non_numeric_ids = sorted(set(non_numeric_ids))
        if non_numeric_ids:
            start = (max(self.id_to_name.keys()) + 1) if self.id_to_name else 0
            for idx, sid in enumerate(non_numeric_ids):
                # Find a representative name for this ID
                rep_name = sid
                for filename in filenames:
                    parts = filename.split('_')
                    if len(parts) >= 2 and parts[0].strip() == sid:
                        rep_name = parts[1].replace("_", " ").strip()
                        break
                self.id_to_name[start + idx] = rep_name
        
        if len(self.id_to_name) == 0:
            messagebox.showerror("Error", "No valid images found!")
            self.status_label.config(text="Error: No valid images", fg="red")
            return
        
        # Second pass: load images and assign labels by ID
        for filename in filenames:
            path = os.path.join(self.training_dir, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or image.size == 0:
                print(f"Warning: Could not load {filename}")
                continue
            
            # Ensure image is the correct size
            if image.shape != self.face_size:
                image = cv2.resize(image, self.face_size)
            
            # Extract student id and get label
            parts = filename.split('_')
            if len(parts) >= 2:
                student_id_str = parts[0].strip()
                label = None
                try:
                    label = int(student_id_str)
                except ValueError:
                    # map non-numeric ids to the labels we assigned above
                    for k, v in self.id_to_name.items():
                        if v.replace(" ", "_").lower() == parts[1].lower():
                            label = k
                            break
                if label is not None and label in self.id_to_name:
                    faces.append(image)
                    labels.append(label)
        
        if len(faces) == 0:
            messagebox.showerror("Error", "No valid images found!")
            self.status_label.config(text="Error: No valid images", fg="red")
            return
        
        if len(set(labels)) < 2:
            messagebox.showwarning("Warning", f"Only one person found in training data. Recognition may not work properly.")
        
        try:
            # Train the recognizer
            self.recognizer.train(faces, np.array(labels, dtype=np.int32))
            
            # Save model
            os.makedirs(os.path.join(self.project_dir, "data"), exist_ok=True)
            trainer_path = os.path.join(self.project_dir, "data", "trainer.yml")
            self.recognizer.save(trainer_path)
            
            # Save label mappings (label/id -> name)
            mapping_path = os.path.join(self.project_dir, "data", "label_mappings.txt")
            with open(mapping_path, 'w', encoding='utf-8') as f:
                for label in sorted(self.id_to_name.keys()):
                    f.write(f"{label}:{self.id_to_name[label]}\n")
            
            self.trained = True
            
            info_msg = f"Model trained successfully!\n\n"
            info_msg += f"Students enrolled: {len(set(labels))}\n"
            info_msg += f"Total images: {len(faces)}\n\n"
            info_msg += "Students:\n"
            for label in sorted(self.id_to_name.keys()):
                count = labels.count(label)
                info_msg += f"  - {self.id_to_name[label]} (ID: {label}): {count} images\n"
            
            messagebox.showinfo("Success", info_msg)
            self.status_label.config(text=f"Model trained: {len(set(labels))} students", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_label.config(text="Training failed", fg="red")
            self.trained = False
    
    def mark_attendance(self):
        if not self.trained:
            messagebox.showerror("Error", "Train model first!")
            return
        
        # For reliable "unknown" rejection, we need at least 2 different
        # enrolled students; with only 1 person in the model, the recognizer
        # tends to map every face to that single label.
        if not self.id_to_name:
            messagebox.showerror("Error", "Model not properly trained! Train again.")
            return
        if len(self.id_to_name) < 2:
            messagebox.showerror(
                "Not Enough Students",
                "Please enroll and train at least 2 different students before "
                "marking attendance.\n\nWith only 1 person trained, the system "
                "cannot reliably treat other faces as 'Unknown'."
            )
            return
        
        # Subject selection
        subject = self.select_subject()
        if not subject:
            return
        
        self.status_label.config(text=f"Marking attendance for {subject}...", fg="orange")
        self.root.update()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return
        
        # Apply camera settings
        self.apply_camera_settings(cap)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        attendance = {}
        last_detection_time = {}
        # Require a stable streak of strong matches before marking present.
        # This reduces false positives when an unknown face appears.
        required_streak = 6
        match_streaks = {}  # name -> current streak count
        
        self.status_label.config(text=f"Detecting faces... (Press Q to quit)", fg="blue")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(110, 110)
            )
            
            current_time = time.time()
            
            for (x, y, w, h) in faces:
                # Extract and preprocess face ROI
                face_roi = gray[y:y+h, x:x+w]
                face_processed = self.preprocess_face(face_roi)
                
                # Predict with confidence
                label, confidence = self.recognizer.predict(face_processed)
                
                # LBPH: Lower confidence = better match (distance).
                # Use configurable threshold + require several detections.
                if confidence <= self.recognition_threshold and label in self.id_to_name:
                    name = self.id_to_name[label]

                    # Increase streak for this person; reset others gradually (optional)
                    match_streaks[name] = match_streaks.get(name, 0) + 1

                    # Only mark attendance after a strong streak, and not more than once every 2 seconds
                    if (
                        match_streaks[name] >= required_streak
                        and (
                            name not in last_detection_time
                            or (current_time - last_detection_time[name]) >= 2.0
                        )
                    ):
                        attendance[name] = "Present"
                        last_detection_time[name] = current_time
                        self.status_label.config(text=f"Detected: {name}", fg="green")

                    # Draw green rectangle for recognized face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, f"{name}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f"{int(confidence)}", (x, y+h+25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif confidence <= (self.recognition_threshold + 15) and label in self.id_to_name:
                    # Uncertain match - yellow
                    name = self.id_to_name[label]
                    # Uncertain match should not build streak; reset to avoid false positives
                    match_streaks[name] = 0
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, f"{name}?", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Unknown face - red
                    # Unknown should not build streaks for any label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display attendance count
            cv2.putText(frame, f"Attendance: {len(attendance)}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f'Marking Attendance - {subject} (Press Q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if attendance:
            self.save_attendance(subject, attendance)
            self.status_label.config(text=f"Attendance saved: {len(attendance)} students", fg="green")
        else:
            messagebox.showinfo("Info", "No attendance recorded.")
            self.status_label.config(text="Ready", fg="green")
    
    # NOTE: The previous 'mark_all_attendance' (single-photo group attendance with
    # class roster and absent list) has been removed to keep the app simpler.
    # Now there is a single clear way to mark attendance: the 'Mark Attendance'
    # button using the live camera stream above.
    
    def save_attendance(self, subject, data):
        import pandas as pd
        from datetime import datetime
        os.makedirs(os.path.join(self.project_dir, "data"), exist_ok=True)
        
        # Create DataFrame with proper columns
        attendance_list = []
        for name, status in data.items():
            attendance_list.append({
                'Name': name,
                'Status': status,
                'Time': datetime.now().strftime("%H:%M:%S"),
                'Date': datetime.now().strftime("%Y-%m-%d")
            })
        
        df = pd.DataFrame(attendance_list)
        filename = f"{self.project_dir}/data/{subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        messagebox.showinfo("Success", f"Attendance saved!\n\nFile: {os.path.basename(filename)}\nStudents: {len(attendance_list)}")
    
    def delete_student(self):
        """Delete a student's record - removes all training images and updates model"""
        # Get all enrolled students
        if not os.path.exists(self.training_dir):
            messagebox.showinfo("Info", "No students enrolled yet!")
            return
        
        # Get all image files
        all_files = os.listdir(self.training_dir)
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            messagebox.showinfo("Info", "No students enrolled yet!")
            return
        
        # Extract unique student names
        student_names = set()
        student_info = {}  # name -> {id, image_count}
        
        for filename in image_files:
            parts = filename.split('_')
            if len(parts) >= 2:
                student_id = parts[0]
                student_name = parts[1]
                student_names.add(student_name)
                
                if student_name not in student_info:
                    student_info[student_name] = {'id': student_id, 'count': 0}
                student_info[student_name]['count'] += 1
        
        if not student_names:
            messagebox.showinfo("Info", "No valid students found!")
            return
        
        # Create delete window
        delete_win = tk.Toplevel(self.root)
        delete_win.title("Delete Student Record")
        delete_win.geometry("500x400")
        delete_win.configure(bg="#f0f0f0")
        
        # Header
        header = tk.Label(delete_win, text="Delete Student Record", 
                         font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#e74c3c")
        header.pack(pady=15)
        
        # Instructions
        tk.Label(delete_win, text="Select a student to delete:", 
                font=("Arial", 10), bg="#f0f0f0").pack(pady=5)
        
        # Listbox with scrollbar
        list_frame = tk.Frame(delete_win, bg="#f0f0f0")
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        student_listbox = tk.Listbox(list_frame, font=("Arial", 11), 
                                     yscrollcommand=scrollbar.set, height=12)
        student_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=student_listbox.yview)
        
        # Populate listbox with student information
        sorted_students = sorted(student_names)
        for name in sorted_students:
            info = student_info[name]
            display_text = f"{name} (ID: {info['id']}) - {info['count']} image(s)"
            student_listbox.insert(tk.END, display_text)
        
        # Select first item
        if sorted_students:
            student_listbox.selection_set(0)
        
        # Info label
        info_label = tk.Label(delete_win, 
                             text="⚠ Warning: This will delete all training images for the selected student.", 
                             font=("Arial", 9), bg="#f0f0f0", fg="#e67e22", wraplength=450)
        info_label.pack(pady=5)
        
        # Buttons frame
        btn_frame = tk.Frame(delete_win, bg="#f0f0f0")
        btn_frame.pack(pady=15)
        
        def confirm_delete():
            selection = student_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a student to delete!")
                return
            
            selected_index = selection[0]
            selected_name = sorted_students[selected_index]
            info = student_info[selected_name]
            
            # Confirm deletion
            confirm_msg = f"Are you sure you want to delete student:\n\n"
            confirm_msg += f"Name: {selected_name}\n"
            confirm_msg += f"ID: {info['id']}\n"
            confirm_msg += f"Training Images: {info['count']}\n\n"
            confirm_msg += "This action cannot be undone!"
            
            if not messagebox.askyesno("Confirm Deletion", confirm_msg, icon="warning"):
                return
            
            # Delete all images for this student
            deleted_count = 0
            failed_count = 0
            
            self.status_label.config(text=f"Deleting {selected_name}...", fg="orange")
            delete_win.update()
            
            for filename in image_files:
                parts = filename.split('_')
                if len(parts) >= 2 and parts[1] == selected_name:
                    file_path = os.path.join(self.training_dir, filename)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Failed to delete {filename}: {e}")
                        failed_count += 1
            
            # Show results
            result_msg = f"Deletion completed!\n\n"
            result_msg += f"Deleted: {deleted_count} image(s)\n"
            if failed_count > 0:
                result_msg += f"Failed: {failed_count} image(s)\n"
            result_msg += f"\nThe model needs to be retrained."
            
            messagebox.showinfo("Deletion Complete", result_msg)
            
            # Mark model as untrained since student data changed
            self.trained = False
            self.label_dict = {}
            self.id_to_name = {}
            
            # Option to retrain immediately
            if messagebox.askyesno("Retrain Model", "Do you want to retrain the model now?"):
                delete_win.destroy()
                self.train_model()
            else:
                delete_win.destroy()
                self.status_label.config(text=f"Student deleted. Please retrain model.", fg="orange")
        
        def cancel_delete():
            delete_win.destroy()
        
        tk.Button(btn_frame, text="Delete Selected", command=confirm_delete, 
                 bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=cancel_delete, 
                 bg="#95a5a6", fg="white", font=("Arial", 11), 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
    
    def load_trained_model(self):
        """Load previously trained model if it exists"""
        trainer_path = os.path.join(self.project_dir, "data", "trainer.yml")
        mapping_path = os.path.join(self.project_dir, "data", "label_mappings.txt")
        
        if os.path.exists(trainer_path):
            try:
                self.recognizer.read(trainer_path)
                
                # Try to load label mappings from file (label/id -> name)
                if os.path.exists(mapping_path):
                    self.id_to_name = {}
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if ':' in line:
                                label_id, name = line.strip().split(':', 1)
                                try:
                                    label_id = int(label_id)
                                except ValueError:
                                    continue
                                self.id_to_name[label_id] = name
                else:
                    # Fallback: rebuild from training images using ID from filename
                    self.id_to_name = {}
                    if os.path.exists(self.training_dir):
                        filenames = sorted(os.listdir(self.training_dir))
                        non_numeric_ids = set()
                        for filename in filenames:
                            parts = filename.split('_')
                            if len(parts) >= 2:
                                sid = parts[0].strip()
                                sname = parts[1].replace("_", " ").strip()
                                try:
                                    label = int(sid)
                                    if label not in self.id_to_name:
                                        self.id_to_name[label] = sname
                                except ValueError:
                                    non_numeric_ids.add(sid)

                        # Assign labels for non-numeric IDs (deterministic order)
                        if non_numeric_ids:
                            start = (max(self.id_to_name.keys()) + 1) if self.id_to_name else 0
                            for idx, sid in enumerate(sorted(non_numeric_ids)):
                                rep_name = sid
                                for filename in filenames:
                                    parts = filename.split('_')
                                    if len(parts) >= 2 and parts[0].strip() == sid:
                                        rep_name = parts[1].replace("_", " ").strip()
                                        break
                                self.id_to_name[start + idx] = rep_name
                
                if self.id_to_name:
                    self.trained = True
                    self.status_label.config(text=f"Model loaded: {len(self.id_to_name)} students", fg="green")
                    
            except Exception as e:
                print(f"Could not load model: {e}")
                self.trained = False
    
    def view_attendance(self):
        """View attendance with present and absent students"""
        import pandas as pd
        from datetime import datetime
        
        data_dir = os.path.join(self.project_dir, "data")
        if not os.path.exists(data_dir):
            messagebox.showinfo("Info", "No attendance files yet!")
            return
        
        # Get all Excel files
        all_files = os.listdir(data_dir)
        excel_files = [f for f in all_files if f.lower().endswith('.xlsx')]
        
        if not excel_files:
            messagebox.showinfo("Info", "No attendance files yet!")
            return
        
        # Sort files by date (newest first)
        excel_files.sort(reverse=True)
        
        # Create selection window
        select_win = tk.Toplevel(self.root)
        select_win.title("Select Attendance File")
        select_win.geometry("600x400")
        select_win.configure(bg="#f0f0f0")
        
        tk.Label(select_win, text="Select Attendance File to View", 
                font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#667eea").pack(pady=15)
        
        # Listbox with scrollbar
        list_frame = tk.Frame(select_win, bg="#f0f0f0")
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        file_listbox = tk.Listbox(list_frame, font=("Arial", 10), 
                                 yscrollcommand=scrollbar.set, height=12)
        file_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=file_listbox.yview)
        
        # Store file paths
        file_paths = {}
        for idx, filename in enumerate(excel_files):
            file_listbox.insert(tk.END, filename)
            file_paths[idx] = os.path.join(data_dir, filename)
        
        if excel_files:
            file_listbox.selection_set(0)
        
        def show_attendance():
            selection = file_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a file!")
                return
            
            selected_index = selection[0]
            if selected_index not in file_paths:
                messagebox.showerror("Error", "File not found!")
                return
            
            file_path = file_paths[selected_index]
            select_win.destroy()
            
            try:
                # Read Excel file
                df = pd.read_excel(file_path)
                
                # Get present students (assuming 'Name' column exists)
                if 'Name' not in df.columns:
                    messagebox.showerror("Error", "Invalid attendance file format!")
                    return
                
                present_names = set(df['Name'].str.strip().str.lower())
                
                # Get all enrolled students from training images
                enrolled_students = {}
                if os.path.exists(self.training_dir):
                    for filename in os.listdir(self.training_dir):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            parts = filename.split('_')
                            if len(parts) >= 2:
                                student_id = parts[0]
                                student_name = parts[1].replace("_", " ")
                                if student_id not in enrolled_students:
                                    enrolled_students[student_id] = student_name
                
                if not enrolled_students:
                    messagebox.showinfo("Info", "No enrolled students found!")
                    return
                
                # Calculate absent students
                absent_students = []
                for student_id, student_name in enrolled_students.items():
                    if student_name.lower() not in present_names:
                        absent_students.append({'id': student_id, 'name': student_name})
                
                # Create view window
                view_win = tk.Toplevel(self.root)
                view_win.title(f"Attendance View - {os.path.basename(file_path)}")
                view_win.geometry("800x600")
                view_win.configure(bg="#f0f0f0")
                
                # Header
                header = tk.Label(view_win, text="Attendance Report", 
                                 font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#667eea")
                header.pack(pady=15)
                
                # File info
                file_info = tk.Label(view_win, text=f"File: {os.path.basename(file_path)}", 
                                    font=("Arial", 10), bg="#f0f0f0", fg="#555")
                file_info.pack(pady=5)
                
                # Create notebook for tabs
                notebook = ttk.Notebook(view_win)
                notebook.pack(fill="both", expand=True, padx=20, pady=10)
                
                # Present tab
                present_frame = tk.Frame(notebook, bg="#f0f0f0")
                notebook.add(present_frame, text=f"✅ Present ({len(present_names)})")
                
                present_list_frame = tk.Frame(present_frame, bg="#f0f0f0")
                present_list_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                present_scrollbar = tk.Scrollbar(present_list_frame)
                present_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                present_listbox = tk.Listbox(present_list_frame, font=("Arial", 11), 
                                            yscrollcommand=present_scrollbar.set, 
                                            bg="white", selectmode=tk.SINGLE)
                present_listbox.pack(side=tk.LEFT, fill="both", expand=True)
                present_scrollbar.config(command=present_listbox.yview)
                
                # Add present students (sorted) with IDs
                present_students_list = []
                for name in sorted(present_names):
                    # Find original name (case-sensitive) and ID from enrolled students
                    for sid, sname in enrolled_students.items():
                        if sname.lower() == name:
                            present_students_list.append({'id': sid, 'name': sname})
                            break
                
                # Sort by name and display with ID
                for student in sorted(present_students_list, key=lambda x: x['name']):
                    present_listbox.insert(tk.END, f"{student['name']} (ID: {student['id']})")
                
                # Absent tab
                absent_frame = tk.Frame(notebook, bg="#f0f0f0")
                notebook.add(absent_frame, text=f"❌ Absent ({len(absent_students)})")
                
                absent_list_frame = tk.Frame(absent_frame, bg="#f0f0f0")
                absent_list_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                absent_scrollbar = tk.Scrollbar(absent_list_frame)
                absent_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                absent_listbox = tk.Listbox(absent_list_frame, font=("Arial", 11), 
                                           yscrollcommand=absent_scrollbar.set, 
                                           bg="white", selectmode=tk.SINGLE)
                absent_listbox.pack(side=tk.LEFT, fill="both", expand=True)
                absent_scrollbar.config(command=absent_listbox.yview)
                
                # Add absent students (sorted)
                for student in sorted(absent_students, key=lambda x: x['name']):
                    absent_listbox.insert(tk.END, f"{student['name']} (ID: {student['id']})")
                
                # Summary label
                summary = tk.Label(view_win, 
                                  text=f"Total Enrolled: {len(enrolled_students)} | "
                                       f"Present: {len(present_names)} | "
                                       f"Absent: {len(absent_students)}", 
                                  font=("Arial", 10, "bold"), bg="#f0f0f0", fg="#333")
                summary.pack(pady=10)
                
                # Close button
                tk.Button(view_win, text="Close", command=view_win.destroy, 
                         bg="#95a5a6", fg="white", font=("Arial", 11), 
                         width=15, height=1).pack(pady=10)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read attendance file:\n{str(e)}")
        
        def cancel():
            select_win.destroy()
        
        # Buttons
        btn_frame = tk.Frame(select_win, bg="#f0f0f0")
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="View Attendance", command=show_attendance, 
                 bg="#4ecdc4", fg="white", font=("Arial", 11, "bold"), 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=cancel, 
                 bg="#95a5a6", fg="white", font=("Arial", 11), 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
    
    def export_excel(self):
        """View, delete, and export/download Excel attendance files"""
        import shutil
        from datetime import datetime
        
        data_dir = os.path.join(self.project_dir, "data")
        if not os.path.exists(data_dir):
            messagebox.showinfo("Info", "No attendance files yet!")
            return
        
        # Get all Excel files
        all_files = os.listdir(data_dir)
        excel_files = [f for f in all_files if f.lower().endswith('.xlsx')]
        
        if not excel_files:
            messagebox.showinfo("Info", "No attendance files yet!")
            return
        
        # Sort files by date (newest first)
        excel_files.sort(reverse=True)
        
        # Create export window
        export_win = tk.Toplevel(self.root)
        export_win.title("View/Export/Delete Excel Files")
        export_win.geometry("700x500")
        export_win.configure(bg="#f0f0f0")
        
        # Header
        header = tk.Label(export_win, text="Attendance Excel Files", 
                         font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#667eea")
        header.pack(pady=15)
        
        # Info label
        info_label = tk.Label(export_win, 
                             text=f"Total files: {len(excel_files)}", 
                             font=("Arial", 10), bg="#f0f0f0")
        info_label.pack(pady=5)
        
        # Listbox with scrollbar
        list_frame = tk.Frame(export_win, bg="#f0f0f0")
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        file_listbox = tk.Listbox(list_frame, font=("Arial", 10), 
                                  yscrollcommand=scrollbar.set, 
                                  selectmode=tk.SINGLE, height=15)
        file_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=file_listbox.yview)
        
        # Store file paths mapping
        file_paths = {}
        
        # Populate listbox with file information
        for filename in excel_files:
            file_path = os.path.join(data_dir, filename)
            try:
                # Get file size and modification time
                file_size = os.path.getsize(file_path)
                mod_time = os.path.getmtime(file_path)
                mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                
                # Format file size
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
                display_text = f"{filename} | {size_str} | {mod_date}"
                file_listbox.insert(tk.END, display_text)
                file_paths[file_listbox.size() - 1] = file_path
            except Exception as e:
                print(f"Error reading file info for {filename}: {e}")
        
        # Select first item
        if excel_files:
            file_listbox.selection_set(0)
        
        # Buttons frame
        btn_frame = tk.Frame(export_win, bg="#f0f0f0")
        btn_frame.pack(pady=15)
        
        def export_selected():
            """Export/download selected file"""
            selection = file_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a file to export!")
                return
            
            selected_index = selection[0]
            if selected_index not in file_paths:
                messagebox.showerror("Error", "Selected file not found!")
                return
            
            source_path = file_paths[selected_index]
            filename = os.path.basename(source_path)
            
            # Ask user where to save the file
            destination = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=filename,
                title="Save Excel File As..."
            )
            
            if destination:
                try:
                    shutil.copy2(source_path, destination)
                    messagebox.showinfo("Success", f"File exported successfully!\n\nSaved to:\n{destination}")
                    self.status_label.config(text=f"File exported: {filename}", fg="green")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export file:\n{str(e)}")
        
        def delete_selected():
            """Delete selected file"""
            selection = file_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a file to delete!")
                return
            
            selected_index = selection[0]
            if selected_index not in file_paths:
                messagebox.showerror("Error", "Selected file not found!")
                return
            
            file_path = file_paths[selected_index]
            filename = os.path.basename(file_path)
            
            # Confirm deletion
            confirm_msg = f"Are you sure you want to delete:\n\n{filename}\n\nThis action cannot be undone!"
            
            if not messagebox.askyesno("Confirm Deletion", confirm_msg, icon="warning"):
                return
            
            try:
                os.remove(file_path)
                messagebox.showinfo("Success", f"File deleted successfully!\n\n{filename}")
                
                # Refresh the list
                export_win.destroy()
                self.export_excel()  # Reload the window
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete file:\n{str(e)}")
        
        def view_selected():
            """Open selected file with default application"""
            selection = file_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a file to view!")
                return
            
            selected_index = selection[0]
            if selected_index not in file_paths:
                messagebox.showerror("Error", "Selected file not found!")
                return
            
            file_path = file_paths[selected_index]
            
            try:
                # Try to open the file with the default application
                if os.name == 'nt':  # Windows
                    os.startfile(file_path)
                elif os.name == 'posix':  # macOS and Linux
                    import subprocess
                    subprocess.call(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', file_path])
                else:
                    messagebox.showinfo("Info", f"File location:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file:\n{str(e)}\n\nFile location: {file_path}")
        
        # Buttons
        tk.Button(btn_frame, text="📄 View", command=view_selected, 
                 bg="#4ecdc4", fg="white", font=("Arial", 10, "bold"), 
                 width=12, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="💾 Export/Download", command=export_selected, 
                 bg="#667eea", fg="white", font=("Arial", 10, "bold"), 
                 width=18, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="🗑️ Delete", command=delete_selected, 
                 bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), 
                 width=12, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Close", command=export_win.destroy, 
                 bg="#95a5a6", fg="white", font=("Arial", 10), 
                 width=12, height=2).pack(side=tk.LEFT, padx=5)
    
    def select_subject(self):
        """Select subject from list or enter new one"""
        if not self.subjects:
            # No subjects saved, ask for input
            subject = simpledialog.askstring("Subject", "Enter Subject Name:")
            return subject
        
        # Create subject selection window
        select_win = tk.Toplevel(self.root)
        select_win.title("Select Subject")
        select_win.geometry("400x250")
        select_win.configure(bg="#f0f0f0")
        
        selected_subject = [None]  # Use list to modify from nested function
        
        tk.Label(select_win, text="Select Subject:", font=("Arial", 12, "bold"), 
                bg="#f0f0f0").pack(pady=15)
        
        # Listbox for subjects
        list_frame = tk.Frame(select_win, bg="#f0f0f0")
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        subject_listbox = tk.Listbox(list_frame, font=("Arial", 11), height=8)
        subject_listbox.pack(fill="both", expand=True)
        
        for subject in self.subjects:
            subject_listbox.insert(tk.END, subject)
        
        if self.subjects:
            subject_listbox.selection_set(0)
        
        def on_select():
            selection = subject_listbox.curselection()
            if selection:
                selected_subject[0] = self.subjects[selection[0]]
            select_win.destroy()
        
        def on_new():
            new_subject = simpledialog.askstring("New Subject", "Enter Subject Name:")
            if new_subject and new_subject.strip():
                new_subject = new_subject.strip()
                if new_subject not in self.subjects:
                    self.subjects.append(new_subject)
                    self.save_subjects()
                selected_subject[0] = new_subject
            select_win.destroy()
        
        def on_cancel():
            select_win.destroy()
        
        btn_frame = tk.Frame(select_win, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Select", command=on_select, 
                 bg="#667eea", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="New Subject", command=on_new, 
                 bg="#38ef7d", fg="white", width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel, 
                 bg="#95a5a6", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        
        select_win.wait_window()
        return selected_subject[0]
    
    def open_settings(self):
        """Open settings window for camera and recognition settings"""
        settings_win = tk.Toplevel(self.root)
        settings_win.title("System Settings")
        settings_win.geometry("500x500")
        settings_win.configure(bg="#f0f0f0")
        
        # Header
        header = tk.Label(settings_win, text="System Settings", 
                         font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#667eea")
        header.pack(pady=15)
        
        # Notebook for tabs
        notebook = ttk.Notebook(settings_win)
        notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Camera Settings Tab
        camera_frame = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(camera_frame, text="📷 Camera Settings")
        
        tk.Label(camera_frame, text="Camera Resolution", font=("Arial", 11, "bold"), 
                bg="#f0f0f0").pack(pady=(20, 5))
        
        resolution_frame = tk.Frame(camera_frame, bg="#f0f0f0")
        resolution_frame.pack(pady=10)
        
        tk.Label(resolution_frame, text="Width:", bg="#f0f0f0").grid(row=0, column=0, padx=5, pady=5)
        width_var = tk.IntVar(value=self.camera_settings['width'])
        width_entry = tk.Entry(resolution_frame, textvariable=width_var, width=10)
        width_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(resolution_frame, text="Height:", bg="#f0f0f0").grid(row=0, column=2, padx=5, pady=5)
        height_var = tk.IntVar(value=self.camera_settings['height'])
        height_entry = tk.Entry(resolution_frame, textvariable=height_var, width=10)
        height_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Common resolutions
        common_res = tk.Frame(camera_frame, bg="#f0f0f0")
        common_res.pack(pady=5)
        tk.Label(common_res, text="Quick select:", bg="#f0f0f0", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        def set_res(w, h):
            width_var.set(w)
            height_var.set(h)
        
        tk.Button(common_res, text="640x480", command=lambda: set_res(640, 480), width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(common_res, text="1280x720", command=lambda: set_res(1280, 720), width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(common_res, text="1920x1080", command=lambda: set_res(1920, 1080), width=8).pack(side=tk.LEFT, padx=2)
        
        tk.Label(camera_frame, text="Brightness (-64 to 64)", font=("Arial", 11, "bold"), 
                bg="#f0f0f0").pack(pady=(20, 5))
        brightness_var = tk.IntVar(value=self.camera_settings['brightness'])
        brightness_scale = tk.Scale(camera_frame, from_=-64, to=64, orient=tk.HORIZONTAL, 
                                    variable=brightness_var, bg="#f0f0f0", length=300)
        brightness_scale.pack(pady=5)
        
        tk.Label(camera_frame, text="Contrast (0 to 64)", font=("Arial", 11, "bold"), 
                bg="#f0f0f0").pack(pady=(20, 5))
        contrast_var = tk.IntVar(value=self.camera_settings['contrast'])
        contrast_scale = tk.Scale(camera_frame, from_=0, to=64, orient=tk.HORIZONTAL, 
                                  variable=contrast_var, bg="#f0f0f0", length=300)
        contrast_scale.pack(pady=5)
        
        # Recognition Settings Tab
        recognition_frame = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(recognition_frame, text="🎯 Recognition Settings")
        
        tk.Label(recognition_frame, text="Recognition Threshold", font=("Arial", 12, "bold"), 
                bg="#f0f0f0").pack(pady=(30, 10))
        tk.Label(recognition_frame, text="Lower values = stricter (fewer false positives)\nHigher values = more lenient (more matches)", 
                bg="#f0f0f0", font=("Arial", 9), justify=tk.CENTER).pack(pady=5)
        
        tk.Label(recognition_frame, text="Attendance Threshold (0-100):", 
                font=("Arial", 10), bg="#f0f0f0").pack(pady=(20, 5))
        regular_threshold_var = tk.IntVar(value=self.recognition_threshold)
        regular_scale = tk.Scale(recognition_frame, from_=30, to=100, orient=tk.HORIZONTAL, 
                                variable=regular_threshold_var, bg="#f0f0f0", length=350)
        regular_scale.pack(pady=5)
        regular_value_label = tk.Label(recognition_frame, text=f"Current: {self.recognition_threshold}", 
                                       bg="#f0f0f0")
        regular_value_label.pack()
        
        def update_regular_label(val):
            regular_value_label.config(text=f"Current: {int(float(val))}")
        regular_scale.config(command=update_regular_label)
        
        # Buttons
        btn_frame = tk.Frame(settings_win, bg="#f0f0f0")
        btn_frame.pack(pady=15)
        
        def save_settings_click():
            try:
                self.camera_settings['width'] = width_var.get()
                self.camera_settings['height'] = height_var.get()
                self.camera_settings['brightness'] = brightness_var.get()
                self.camera_settings['contrast'] = contrast_var.get()
                self.recognition_threshold = regular_threshold_var.get()
                
                self.save_settings()
                messagebox.showinfo("Success", "Settings saved successfully!")
                settings_win.destroy()
                self.status_label.config(text="Settings updated", fg="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")

        def reset_to_defaults():
            """Reset UI controls + app settings back to recommended defaults."""
            if not messagebox.askyesno("Reset to Defaults", "Reset all settings to default values?"):
                return

            # Recommended defaults
            width_var.set(640)
            height_var.set(480)
            brightness_var.set(0)
            contrast_var.set(0)
            regular_threshold_var.set(40)
            update_regular_label(40)

            # Apply to app state and persist to config.json
            self.camera_settings['width'] = 640
            self.camera_settings['height'] = 480
            self.camera_settings['brightness'] = 0
            self.camera_settings['contrast'] = 0
            self.recognition_threshold = 40
            self.save_settings()

            messagebox.showinfo("Defaults Restored", "Default settings restored and saved.")
            self.status_label.config(text="Settings reset to defaults", fg="green")
        
        tk.Button(btn_frame, text="Save Settings", command=save_settings_click, 
                 bg="#667eea", fg="white", font=("Arial", 11, "bold"), width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Reset to Defaults", command=reset_to_defaults, 
                 bg="#e67e22", fg="white", font=("Arial", 11, "bold"), width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=settings_win.destroy, 
                 bg="#95a5a6", fg="white", font=("Arial", 11), width=15).pack(side=tk.LEFT, padx=5)
    
    def manage_subjects(self):
        """Manage subjects - add, edit, delete"""
        manage_win = tk.Toplevel(self.root)
        manage_win.title("Manage Subjects")
        manage_win.geometry("500x400")
        manage_win.configure(bg="#f0f0f0")
        
        # Header
        header = tk.Label(manage_win, text="Manage Subjects", 
                         font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#3498db")
        header.pack(pady=15)
        
        # Listbox with scrollbar
        list_frame = tk.Frame(manage_win, bg="#f0f0f0")
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        subject_listbox = tk.Listbox(list_frame, font=("Arial", 11), 
                                     yscrollcommand=scrollbar.set, height=12)
        subject_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=subject_listbox.yview)
        
        def refresh_list():
            subject_listbox.delete(0, tk.END)
            for subject in self.subjects:
                subject_listbox.insert(tk.END, subject)
            if self.subjects:
                subject_listbox.selection_set(0)
        
        refresh_list()
        
        # Entry for new/edit
        entry_frame = tk.Frame(manage_win, bg="#f0f0f0")
        entry_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(entry_frame, text="Subject Name:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        subject_entry = tk.Entry(entry_frame, width=25)
        subject_entry.pack(side=tk.LEFT, padx=5)
        
        def add_subject():
            name = subject_entry.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a subject name!")
                return
            if name in self.subjects:
                messagebox.showwarning("Warning", "Subject already exists!")
                return
            self.subjects.append(name)
            self.save_subjects()
            refresh_list()
            subject_entry.delete(0, tk.END)
            self.status_label.config(text=f"Subject '{name}' added", fg="green")
        
        def edit_subject():
            selection = subject_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a subject to edit!")
                return
            old_name = self.subjects[selection[0]]
            new_name = subject_entry.get().strip()
            if not new_name:
                messagebox.showwarning("Warning", "Please enter a new subject name!")
                return
            if new_name in self.subjects and new_name != old_name:
                messagebox.showwarning("Warning", "Subject name already exists!")
                return
            self.subjects[selection[0]] = new_name
            self.save_subjects()
            refresh_list()
            subject_entry.delete(0, tk.END)
            self.status_label.config(text=f"Subject updated to '{new_name}'", fg="green")
        
        def delete_subject():
            selection = subject_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a subject to delete!")
                return
            name = self.subjects[selection[0]]
            if messagebox.askyesno("Confirm", f"Delete subject '{name}'?"):
                self.subjects.pop(selection[0])
                self.save_subjects()
                refresh_list()
                subject_entry.delete(0, tk.END)
                self.status_label.config(text=f"Subject '{name}' deleted", fg="orange")
        
        def load_selected():
            selection = subject_listbox.curselection()
            if selection:
                subject_entry.delete(0, tk.END)
                subject_entry.insert(0, self.subjects[selection[0]])
        
        subject_listbox.bind('<Double-Button-1>', lambda e: load_selected())
        
        # Buttons
        btn_frame = tk.Frame(manage_win, bg="#f0f0f0")
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Add", command=add_subject, 
                 bg="#38ef7d", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Edit", command=edit_subject, 
                 bg="#667eea", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Delete", command=delete_subject, 
                 bg="#e74c3c", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Close", command=manage_win.destroy, 
                 bg="#95a5a6", fg="white", width=10).pack(side=tk.LEFT, padx=5)
    
    # Note: the dedicated 'Class Roster' management window has been removed.
    # Now students are defined only through 'Enroll New Student', so there is
    # just one clear place to manage who is in the system.

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()
