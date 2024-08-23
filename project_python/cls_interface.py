import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from cls_deepface import DeepFaceRecognition
from cls_lbph_opencv import LBPHFaceRecognition
from cls_facerecognition import FaceRecognition
from cls_dnn_opencv import DNNRecognition
from cls_insightface import InsightFaceRecognition
from cls_mtcnn import MTCNNFaceRecognition

import threading
from tkinter import simpledialog, messagebox, filedialog

class FaceRecognitionApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.recognition_thread = None
        self.stop_event = threading.Event()  # Используем Event для остановки потока

        self.app_face = FaceRecognition(db_path='face_rec.db')
        self.app_lbph = LBPHFaceRecognition(db_path='lbph.db')
        self.app_dnn = DNNRecognition(db_path='dnn.db')
        self.app_insface = InsightFaceRecognition(db_path='insface.db')
        self.app_deepface =  DeepFaceRecognition(db_path='deepface.db')
        self.app_mtcnn =  MTCNNFaceRecognition(db_path='mtcnn.db')
        self.configure(bg="lightblue")  # Установка фона для окна
        self.geometry("180x750")  # Устанавливаем размер окна 800x600 пикселей
 
        self.title("Face Recognition App")

        self.label_method = tk.Label(self, text="Choose Method:").pack(pady=10)

        self.method_var = tk.StringVar(value="FACEREC")
        tk.Radiobutton(self, text="FaceRecognition", variable=self.method_var, value="FACEREC").pack(pady=5)
        tk.Radiobutton(self, text="LBPH", variable=self.method_var, value="LBPH").pack(pady=5)
        tk.Radiobutton(self, text="DNN", variable=self.method_var, value="DNN").pack(pady=5)
        tk.Radiobutton(self, text="Insightface", variable=self.method_var, value="INSFACE").pack(pady=5)
        tk.Radiobutton(self, text="Deepface", variable=self.method_var, value="DEEPFACE").pack(pady=5)
        tk.Radiobutton(self, text="MTCNN", variable=self.method_var, value="MTCNN").pack(pady=5)

        tk.Label(self, text="Choose Source:").pack(pady=10)

        self.source_var = tk.StringVar(value="Camera")
        tk.Radiobutton(self, text="Camera", variable=self.source_var, value="Camera").pack(pady=5)
        tk.Radiobutton(self, text="Video", variable=self.source_var, value="Video").pack(pady=5)

        tk.Button(self, text="Register User", command=self.register_user).pack(pady=10)
        tk.Button(self, text="Recognize Users", command=self.recognize_users).pack(pady=10)
        tk.Button(self, text="Stop Recognize Users", command=self.stop_recognize_users).pack(pady=10)
        tk.Button(self, text="Show All Users", command=self.show_all_users).pack(pady=10)
        tk.Button(self, text="Delete User by ID", command=self.delete_user).pack(pady=10)
        tk.Button(self, text="Delete All Users", command=self.delete_all_users).pack(pady=10)
        tk.Button(self, text="Set Time for ActivateAlive", command=self.activateAlive).pack(pady=10)
        tk.Button(self, text="Stop ActivateAlive", command=self.stopActivateAlive).pack(pady=10)
        tk.Button(self, text="Exit", command=self.quit).pack(pady=10)

        self.recognition_thread = None
    #================================================================================
    def get_recognition_instance(self):
        method = self.method_var.get()
        if method == "FACEREC":
            return self.app_face
        elif method == "LBPH":
            return self.app_lbph
        elif method == "DEEPFACE":
            return self.app_deepface
        elif method == "DNN":
            return self.app_dnn
        elif method == "INSFACE":
            return self.app_insface
        elif method == "MTCNN":
            return self.app_mtcnn
  
    #================================================================================
    def register_user(self):
        app = self.get_recognition_instance()
        source = self.source_var.get()
        if source == "Camera":
            guid = app.register_users_from_camera()
            if guid:
                messagebox.showinfo("Info", f"User registered successfully.")
            else:
                messagebox.showinfo("Info", f"User not registered.")
        elif source == "Video":
            video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
            if video_path:
                guid = app.register_users_from_video(video_path)
                if guid:
                    messagebox.showinfo("Info", f"User registered successfully.")
                else:
                    messagebox.showinfo("Info", f"User not registered.")

    #================================================================================
    def stop_recognize_users(self):
       app = self.get_recognition_instance()
       app.stop_recognition_process()
       if self.recognition_thread and self.recognition_thread.is_alive():
            self.stop_event.set()  # Устанавливаем флаг для остановки текущего потока
            self.recognition_thread.join()  # Ожидаем завершения потока
            self.recognition_thread = None  # Сбрасываем ссылку на поток

    #================================================================================
    def recognize_users(self):
        # Сначала останавливаем текущий поток, если он существует
        self.stop_recognize_users()
        
        # Сбрасываем событие остановки перед запуском нового потока
        self.stop_event.clear()

        # Выбираем метод распознавания
        app = self.get_recognition_instance()
        source = self.source_var.get()
        
        if source == "Camera":
            self.recognition_thread = threading.Thread(target=self.run_recognition_from_camera, args=(app,))
        elif source == "Video":
            video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
            if video_path:
                self.recognition_thread = threading.Thread(target=self.run_recognition_from_video, args=(app, video_path))

        if self.recognition_thread:
            self.recognition_thread.start()

    def run_recognition_from_camera(self, app):
        app.recognize_users_from_camera()

    def run_recognition_from_video(self, app, video_path):
        app.recognize_users_from_video(video_path)

    #================================================================================
    def show_all_users(self):
        app = self.get_recognition_instance()
        users = app.list_users()
        user_list = "\n".join([f"ID: {user[0]}, Guid: {user[1]}" for user in users])
        messagebox.showinfo("All Users", user_list)

    #================================================================================
    def delete_user(self):
        app = self.get_recognition_instance()
        user_id = simpledialog.askinteger("Input", "Enter the user ID to delete:")
        if user_id:
            result = app.delete_user_by_id(user_id)
            messagebox.showinfo("Info", result)
    
    #================================================================================
    def delete_all_users(self):
        app = self.get_recognition_instance()
        result = app.deleteAllFromData()
        messagebox.showinfo("Info", result)        

    #================================================================================
    def activateAlive(self):
        app = self.get_recognition_instance()
        time = simpledialog.askinteger("Input", "Enter the time in sec for definition live people:")
        if time > 0:
            result = app.setAlivePerson(time)
            messagebox.showinfo("Info", result)

    #================================================================================
    def stopActivateAlive(self):
        app = self.get_recognition_instance()
        app.setAlivePerson(0)
