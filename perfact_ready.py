import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import shutil
from gtts import gTTS
from playsound import playsound
import tempfile
from datetime import datetime
import csv
import threading   # for non-blocking TTS

# ======== Config ========
BG_IMG_PATH = "welcome.jpg"   # keep welcome.jpg in same folder
ATTENDANCE_CSV = "attendance.csv"
# ========================

def load_labels():
    labels = {}
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            for line in f:
                if ":" in line:
                    id, name = line.strip().split(":", 1)
                    labels[int(id)] = name
    return labels

# ---------- Non-blocking TTS ----------
def speak_text(text):
    def run_tts():
        try:
            tts = gTTS(text=text)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name
            tts.save(temp_path)
            playsound(temp_path)
            os.remove(temp_path)
        except Exception as e:
            print("TTS Error:", e)
    threading.Thread(target=run_tts, daemon=True).start()

# ---------- Dataset Capture ----------
def generate_dataset():
    name = simpledialog.askstring("Input", "Enter your name:")
    if not name:
        messagebox.showerror("Error", "Name cannot be empty.")
        return

    if not os.path.exists("labels.txt"):
        open("labels.txt", "w").close()

    with open("labels.txt", "r") as f:
        lines = f.readlines()
        existing = [line.strip().split(":", 1)[1] for line in lines if ":" in line]
        user_id = existing.index(name) + 1 if name in existing else len(existing) + 1

    if name not in existing:
        with open("labels.txt", "a") as f:
            f.write(f"{user_id}:{name}\n")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            return img[y:y+h, x:x+w]
        return None

    cap = cv2.VideoCapture(0)
    count = 0
    os.makedirs("data", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face = face_cropped(frame)
        if face is not None:
            count += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_path = f"data/user.{user_id}.{count}.jpg"
            cv2.imwrite(file_path, face)
            cv2.imshow("Capturing...", face)
        if cv2.waitKey(1) == 13 or count == 200:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Done", f"200 face samples saved for {name}.")

# ---------- Train Model ----------
def train_classifier():
    if not os.path.exists("data"):
        messagebox.showerror("Error", "No data found. Capture faces first.")
        return

    faces, ids = [], []
    for file in os.listdir("data"):
        img = Image.open(os.path.join("data", file)).convert("L")
        face_np = np.array(img, "uint8")
        id = int(file.split(".")[1])
        faces.append(face_np)
        ids.append(id)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(ids))
    clf.write("classifier.xml")
    messagebox.showinfo("Training Complete", "Model trained successfully.")

# ---------- Recognize Faces (NO Attendance) ----------
def draw_and_recognize(img, classifier, scaleFactor, minNeighbors, color, clf, labels):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        id, pred = clf.predict(face)
        if pred < 70:   # FIXED recognition logic
            name = labels.get(id, "UNKNOWN")
        else:
            name = "UNKNOWN"
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img

def recognize_faces():
    if not os.path.exists("classifier.xml"):
        messagebox.showerror("Error", "Train the model first.")
        return
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    labels = load_labels()
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_and_recognize(frame, face_cascade, 1.1, 10, (0, 255, 0), clf, labels)
        cv2.imshow("Recognizing (no attendance)", frame)
        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------- Attendance ----------
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    exists = set()
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0 and row[0] == name and row[1] == date:
                    exists.add(name)
    if name not in exists and name != "UNKNOWN":
        with open(ATTENDANCE_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time])
        speak_text(name)

def take_attendance():
    if not os.path.exists("classifier.xml"):
        messagebox.showerror("Error", "Train the model first.")
        return
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    labels = load_labels()
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    spoken = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            id, pred = clf.predict(face)
            if pred < 70:   # FIXED recognition logic
                name = labels.get(id, "UNKNOWN")
            else:
                name = "UNKNOWN"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if name not in spoken and name != "UNKNOWN":
                spoken.add(name)
                mark_attendance(name)
        cv2.imshow("Take Attendance", frame)
        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------- View Today Attendance ----------
def view_today_attendance():
    if not os.path.exists(ATTENDANCE_CSV):
        messagebox.showinfo("Attendance", "No attendance recorded yet.")
        return
    win = tk.Toplevel()
    win.title("Today's Attendance")
    today = datetime.now().strftime("%Y-%m-%d")
    with open(ATTENDANCE_CSV, 'r') as f:
        rows = [row for row in csv.reader(f) if row and row[1] == today]
    tk.Label(win, text=f"Attendance for {today}", font=("Arial", 14, "bold")).pack(pady=10)
    for i, row in enumerate(rows):
        tk.Label(win, text=f"{i+1}. {row[0]} - {row[2]}", font=("Arial", 11)).pack(anchor="w")

# ---------- Other GUI Functions ----------
def view_records():
    win = tk.Toplevel()
    win.title("View Records")
    canvas = tk.Canvas(win)
    scrollbar = tk.Scrollbar(win, command=canvas.yview)
    frame = tk.Frame(canvas)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    canvas.create_window((0, 0), window=frame, anchor="nw")

    def show_all(user_id, name):
        img_win = tk.Toplevel()
        img_win.title(f"{name}'s Samples")
        c = tk.Canvas(img_win)
        s = tk.Scrollbar(img_win, command=c.yview)
        f = tk.Frame(c)
        c.configure(yscrollcommand=s.set)
        c.pack(side="left", fill="both", expand=True)
        s.pack(side="right", fill="y")
        c.create_window((0, 0), window=f, anchor="nw")
        for i in range(1,201):
            path = f"data/user.{user_id}.{i}.jpg"
            if os.path.exists(path):
                img = Image.open(path).resize((100, 100))
                photo = ImageTk.PhotoImage(img)
                lbl = tk.Label(f, image=photo)
                lbl.image = photo
                lbl.grid(row=(i-1)//5, column=(i-1)%5, padx=5, pady=5)
        f.update_idletasks()
        c.configure(scrollregion=c.bbox("all"))

    row = 0
    for user_id, name in load_labels().items():
        img_path = f"data/user.{user_id}.1.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((100, 100))
            photo = ImageTk.PhotoImage(img)
            btn = tk.Button(frame, image=photo, text=name, compound="top",
                            command=lambda uid=user_id, nm=name: show_all(uid, nm))
            btn.image = photo
            btn.grid(row=row//4, column=row%4, padx=10, pady=10)
            row += 1
    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

def view_all_data():
    win = tk.Toplevel()
    win.title("All Captured Data")
    canvas = tk.Canvas(win)
    scrollbar = tk.Scrollbar(win, command=canvas.yview)
    frame = tk.Frame(canvas)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    canvas.create_window((0, 0), window=frame, anchor="nw")
    for i, file in enumerate(os.listdir("data") if os.path.exists("data") else []):
        img = Image.open(f"data/{file}").resize((100, 100))
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(frame, image=photo, text=file, compound="top")
        lbl.image = photo
        lbl.grid(row=i//5, column=i%5, padx=5, pady=5)
    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

def reset_all():
    if messagebox.askyesno("Confirm", "Delete all data?"):
        shutil.rmtree("data", ignore_errors=True)
        for f in ["labels.txt", "classifier.xml", ATTENDANCE_CSV]:
            if os.path.exists(f):
                os.remove(f)
        messagebox.showinfo("Done", "All data deleted.")

def how_it_works():
    info = """
1. Login with: nick / 3171
2. Capture Face Dataset (200 samples)
3. Train model (LBPH)
4. 'Take Attendance' → recognizes + speaks + saves to CSV
5. 'Recognize Faces' → just display (no CSV, no speech)
6. View Records / View All / Today's Attendance
7. Reset All / Exit
    """
    messagebox.showinfo("How It Works", info)

# ---------- Main GUI ----------
def show_main_gui():
    root = tk.Toplevel()
    root.title("Face Recognition System")
    root.geometry("900x700")
    try:
        bg = Image.open(BG_IMG_PATH).resize((900, 700))
        bg_img = ImageTk.PhotoImage(bg)
        tk.Label(root, image=bg_img).place(x=0, y=0, relwidth=1, relheight=1)
        root.bg = bg_img
    except: pass

    tk.Label(root, text="Face Recognition System", font=("Arial", 22, "bold"), bg="white").pack(pady=10)
    tk.Label(root, text="Developed by: WARISH RAJA", font=("Arial", 14, "bold"), bg="white", fg="blue").pack()
    tk.Label(root, text="DEPT. OF COMPUTER SCIENCE ENGINEERING", font=("Arial", 13), bg="white", fg="darkgreen").pack(pady=5)

    frame = tk.Frame(root, bg="white")
    frame.pack(pady=30)
    btn_opts = {"width": 25, "height": 2, "font": ("Arial", 11, "bold"), "padx": 15, "pady": 15}

    tk.Button(frame, text="Capture Face Dataset", bg="violet", command=generate_dataset, **btn_opts).grid(row=0, column=0, padx=10, pady=10)
    tk.Button(frame, text="Train Model", bg="skyblue", command=train_classifier, **btn_opts).grid(row=0, column=1, padx=10, pady=10)
    tk.Button(frame, text="Take Attendance", bg="orange", command=take_attendance, **btn_opts).grid(row=1, column=0, padx=10, pady=10)
    tk.Button(frame, text="Recognize Faces", bg="lightyellow", command=recognize_faces, **btn_opts).grid(row=1, column=1, padx=10, pady=10)
    tk.Button(frame, text="View Records", bg="lightgreen", command=view_records, **btn_opts).grid(row=2, column=0, padx=10, pady=10)
    tk.Button(frame, text="All Data", bg="lightblue", command=view_all_data, **btn_opts).grid(row=2, column=1, padx=10, pady=10)
    tk.Button(frame, text="Today Attendance", bg="lightpink", command=view_today_attendance, **btn_opts).grid(row=3, column=0, padx=10, pady=10)
    tk.Button(frame, text="Reset All", bg="red", fg="white", command=reset_all, **btn_opts).grid(row=3, column=1, padx=10, pady=10)
    tk.Button(root, text="How It Works", width=30, height=2, font=("Arial", 11), bg="lightgray", command=how_it_works).pack(pady=10)
    tk.Button(root, text="Exit", width=20, height=2, font=("Arial", 11), bg="black", fg="white", command=root.quit).pack(pady=10)

# ---------- Login ----------
def login():
    if username_entry.get() == "nick" and password_entry.get() == "3171":
        login_win.destroy()
        show_main_gui()
    else:
        messagebox.showerror("Login Failed", "Invalid credentials.")

login_win = tk.Tk()
login_win.title("Login - Face Recognition")
login_win.geometry("500x370")
login_win.configure(bg="blue")

tk.Label(login_win, text="Login - Face Recognition System", font=("Arial", 20, "bold"), bg="blue", fg="white").pack(pady=25)
form_frame = tk.Frame(login_win, bg="blue")
form_frame.pack(pady=10)
tk.Label(form_frame, text="Username", bg="blue", fg="white", font=("Arial", 13, "bold")).pack(pady=5)
username_entry = tk.Entry(form_frame, font=("Arial", 12)); username_entry.pack()
tk.Label(form_frame, text="Password", bg="blue", fg="white", font=("Arial", 13, "bold")).pack(pady=5)
password_entry = tk.Entry(form_frame, show="*", font=("Arial", 12)); password_entry.pack()
tk.Button(form_frame, text="Login", bg="lightgreen", font=("Arial", 12, "bold"), width=15, command=login).pack(pady=15)
tk.Label(login_win, text="Developed by: Warish Raja", bg="blue", fg="white", font=("Arial", 12, "bold")).pack(side="bottom", pady=5)
tk.Label(login_win, text="Dept: Computer Science Engineering", bg="blue", fg="white", font=("Arial", 12, "bold")).pack(side="bottom", pady=0)

login_win.mainloop()
