import cv2
import torch
import tkinter as tk
import ttkbootstrap as ttk  # Libreria per un look moderno
import speech_recognition as sr
import pyttsx3
import numpy as np
import pyautogui
import threading
import time
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO

# Inizializza il riconoscimento vocale e la sintesi vocale
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Carica il modello YOLOv8 pre-addestrato
model = YOLO("yolov8n.pt")

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Progetto Negrisolo R.I.C.O")
        self.root.geometry("800x600")
        self.root.configure(bg="#2C2F33")

        # Stile personalizzato
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12), padding=10)

        # Frame per il video
        self.frame = Frame(root, bg="#23272A", bd=2, relief="ridge")
        self.frame.pack(pady=10)
        
        self.label = Label(self.frame)
        self.label.pack()
        
        # Etichetta per mostrare gli oggetti rilevati
        self.detected_label = Label(root, text="", fg="#FFFFFF", bg="#2C2F33", font=("Arial", 14))
        self.detected_label.pack(pady=5)
        
        # Pulsanti moderni
        self.start_button = ttk.Button(root, text="Avvia Riconoscimento Webcam", command=self.start_detection)
        self.start_button.pack(pady=5)
        
        self.screen_button = ttk.Button(root, text="Avvia Riconoscimento Schermo", command=self.toggle_screen_detection)
        self.screen_button.pack(pady=5)
        
        self.stop_button = ttk.Button(root, text="Ferma", command=self.stop_detection)
        self.stop_button.pack(pady=5)
        
        self.voice_button = ttk.Button(root, text="Chiedi cosa vede", command=self.start_voice_recognition)
        self.voice_button.pack(pady=5)
        
        self.voice_label = Label(root, text="", fg="#FFD700", bg="#2C2F33", font=("Arial", 12))
        self.voice_label.pack(pady=5)
        
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.screen_running = False
        self.detected_objects = []
    
    def start_detection(self):
        self.running = True
        self.detect_objects()
    
    def toggle_screen_detection(self):
        if self.screen_running:
            self.screen_running = False
        else:
            self.screen_running = True
            self.detect_screen_objects()
    
    def stop_detection(self):
        self.running = False
        self.screen_running = False
        self.cap.release()
        cv2.destroyAllWindows()
    
    def detect_objects(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)
        
        self.root.after(10, self.detect_objects)
    
    def detect_screen_objects(self):
        if not self.screen_running:
            return
        
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.process_frame(frame)
        
        if self.screen_running:
            self.root.after(10, self.detect_screen_objects)
    
    def process_frame(self, frame):
        results = model(frame)[0]
        detected_objects_temp = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls[0])]
            conf = box.conf[0].item()
            detected_objects_temp.append(label)
            
            # Disegna una nuvoletta bianca dietro il testo
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{label} {conf:.2f}"
            text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
            text_x, text_y = x1, y1 - 10
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
            cv2.putText(frame, text, (text_x, text_y), font, 0.5, (0, 0, 0), 2)
        
        time.sleep(1)  # Ritarda la visualizzazione di 1 secondo
        self.detected_objects = detected_objects_temp
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        
        if self.detected_objects:
            objects_seen = ", ".join(set(self.detected_objects))
            self.detected_label.config(text=f"Oggetti rilevati: {objects_seen}", fg="#00FF00")
        else:
            self.detected_label.config(text="Nessun oggetto rilevato", fg="#FF0000")
    
    def start_voice_recognition(self):
        threading.Thread(target=self.voice_recognition, daemon=True).start()

    def voice_recognition(self):
        with sr.Microphone() as source:
            self.voice_label.config(text="Ascolto...", fg="#FFD700")
            engine.say("Dimmi: Cosa vedi?")
            engine.runAndWait()
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                self.voice_label.config(text=f"Hai detto: {command}", fg="#FFFFFF")
                
                if "cosa vedi" in command:
                    if self.detected_objects:
                        objects_seen = ", ".join(set(self.detected_objects))
                        response = f"Vedo: {objects_seen}"
                    else:
                        response = "Non vedo nessun oggetto chiaramente."
                    
                    engine.say(response)
                    engine.runAndWait()
                    self.voice_label.config(text=response, fg="#00FF00")
            except sr.UnknownValueError:
                self.voice_label.config(text="Non ho capito, puoi ripetere?", fg="#FF0000")
                engine.say("Non ho capito, puoi ripetere?")
                engine.runAndWait()

# Avvia l'app
if __name__ == "__main__":
    root = ttk.Window(themename="darkly")  # Usa un tema moderno
    app = ObjectDetectionApp(root)
    root.mainloop()
