import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
import psycopg2
import dlib
from datetime import datetime

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi detektor wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model landmark wajah
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

# Inisialisasi GUI Tkinter
root = tk.Tk()
root.title("Face Detection and Registration")

# Variabel untuk menyimpan data mahasiswa
nama = tk.StringVar()
nim = tk.StringVar()
jurusan = tk.StringVar()

# Variabel untuk menyimpan posisi wajah sebelumnya
prev_face_position = None
movement_threshold = 50  # Ambang batas pergerakan signifikan

# Fungsi untuk menyimpan data mahasiswa ke dalam database PostgreSQL
def save_to_database():
    # Mengambil nilai dari input form
    nama_mhs = nama.get()
    nim_mhs = nim.get()
    jurusan_mhs = jurusan.get()
    
    # Koneksi ke database PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        database="mahasiswa",
        user="postgres",
        password="atio02"
    )
    cursor = conn.cursor()
    
    # Masukkan data mahasiswa ke dalam tabel mahasiswa
    cursor.execute("INSERT INTO mahasiswa (nama, nim, jurusan) VALUES (%s, %s, %s)",
                   (nama_mhs, nim_mhs, jurusan_mhs))
    conn.commit()
    conn.close()
    
    # Notifikasi jika data berhasil dimasukkan ke dalam database
    messagebox.showinfo("Info", "Data berhasil disimpan ke dalam database!")

    # Setelah data tersimpan, tampilkan frame untuk face recognition
    show_face_recognition_frame()

# Fungsi untuk menampilkan frame kamera dan deteksi wajah
def show_frame():
    global panel  # Membuat panel sebagai variabel global
    panel = tk.Label(root)
    panel.pack(padx=10, pady=10)
    
    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Tampilkan tanggal di pojok kanan atas
            tanggal = datetime.now().strftime("%d-%m-%Y")
            text_size = cv2.getTextSize(tanggal, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(frame, tanggal, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Tampilkan tulisan "Bina Sarana Informatika" di pojok kiri atas
            cv2.putText(frame, "Bina Sarana Informatika", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Tampilkan nama, nim, dan jurusan di pojok kiri atas, di bawah "Bina Sarana Informatika"
                text = f"Nama: {nama.get()}\nNIM: {nim.get()}\nJurusan: {jurusan.get()}"
                text_lines = text.split('\n')
                y_pos = y + 30
                for line in text_lines:
                    cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    y_pos += 30
            
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            panel.imgtk = imgtk
            panel.config(image=imgtk)
            panel.after(10, update_frame)
    
    update_frame()

# Fungsi untuk menampilkan frame untuk face recognition setelah data tersimpan
def show_face_recognition_frame():
    global panel_recognition
    root.title("Face Recognition")
    
    # Menghapus panel input sebelumnya
    for widget in root.winfo_children():
        widget.destroy()
    
    # Panel baru untuk menampilkan frame kamera
    panel_recognition = tk.Label(root)
    panel_recognition.pack(padx=10, pady=10)
    
    # Fungsi untuk menampilkan frame kamera dan deteksi wajah
    def show_recognition_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = detect_faces(frame, face_cascade, predictor)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            panel_recognition.imgtk = imgtk
            panel_recognition.config(image=imgtk)
        panel_recognition.after(10, show_recognition_frame)
    
    # Memulai fungsi untuk menampilkan frame kamera
    show_recognition_frame()

# Fungsi untuk mendeteksi wajah dari frame
def detect_faces(frame, face_cascade, predictor):
    global prev_face_position
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Tampilkan tanggal di pojok kanan atas
    tanggal = datetime.now().strftime("%d-%m-%Y")
    text_size = cv2.getTextSize(tanggal, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(frame, tanggal, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Tampilkan tulisan "Bina Sarana Informatika" di pojok kiri atas
    cv2.putText(frame, "Bina Sarana Informatika", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    for (x, y, w, h) in faces:
        face_rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = predictor(gray, face_rect)
        
        # Mengecek pergerakan wajah
        current_face_position = (x + w//2, y + h//2)
        
        if prev_face_position is not None:
            movement = ((current_face_position[0] - prev_face_position[0]) ** 2 + (current_face_position[1] - prev_face_position[1]) ** 2) ** 0.5
            if movement > movement_threshold:
                cv2.putText(frame, "Hayoo ngapain hayoo", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        prev_face_position = current_face_position
        
        # Tampilkan nama, nim, dan jurusan di pojok kiri atas, di bawah "Bina Sarana Informatika"
        text = f"Nama: {nama.get()}\nNIM: {nim.get()}\nJurusan: {jurusan.get()}"
        text_lines = text.split('\n')
        y_pos = y + 50  # Menambah jarak agar tidak menutupi tulisan di atas
        for line in text_lines:
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y_pos += 30
        
        # Kotak mengikuti wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

# Label dan Entry untuk input nama, nim, dan jurusan
label_nama = tk.Label(root, text="Nama:", font=("Helvetica", 14))
label_nama.pack(anchor=tk.W, padx=10, pady=(20, 5))
entry_nama = tk.Entry(root, textvariable=nama, font=("Helvetica", 12))
entry_nama.pack(anchor=tk.W, padx=10, pady=5)

label_nim = tk.Label(root, text="NIM:", font=("Helvetica", 14))
label_nim.pack(anchor=tk.W, padx=10, pady=5)
entry_nim = tk.Entry(root, textvariable=nim, font=("Helvetica", 12))
entry_nim.pack(anchor=tk.W, padx=10, pady=5)

label_jurusan = tk.Label(root, text="Jurusan:", font=("Helvetica", 14))
label_jurusan.pack(anchor=tk.W, padx=10, pady=5)
entry_jurusan = tk.Entry(root, textvariable=jurusan, font=("Helvetica", 12))
entry_jurusan.pack(anchor=tk.W, padx=10, pady=5)

# Tombol untuk submit data
btn_submit = tk.Button(root, text="Submit", command=save_to_database, font=("Helvetica", 12), bg='#4CAF50', fg='white', relief=tk.FLAT)
btn_submit.pack(pady=20)

# Menutup kamera saat aplikasi ditutup
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
