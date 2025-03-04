import tkinter as tk
from tkinter import ttk, Menu
from tkinter import Frame, Label, Text, Scrollbar, filedialog, messagebox
import cv2
import threading
import time
import pyautogui
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import sqlite3
from datetime import datetime
import os
import json
import logging
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
from functools import lru_cache
import mss  # Per screenshot più veloci
import gc    # Per la gestione della memoria
import speech_recognition as sr
import pyttsx3
import pandas as pd
import plotly.express as px
import pygame
import math
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Configurazione logging
logging.basicConfig(filename='object_detection.log', level=logging.INFO)
# Definisci il dizionario object_uses
object_uses = {
    "person": "Una persona può fare molte cose, come camminare, parlare, ecc.",
    "bicycle": "Una bicicletta è usata per andare in giro.",
    "car": "Un'auto è usata per il trasporto.",
    "bottle": "Una bottiglia è usata per contenere liquidi.",
    "apple": "Una mela è un frutto commestibile.",
    "screw": "Una vite è usata per fissare oggetti insieme.",
    "screwdriver": "Un cacciavite è usato per avvitare o svitare viti.",
    "hammer": "Un martello è usato per battere chiodi.",
    "ruler": "Un righello è usato per misurare lunghezze.",
}

# Aggiungi supporto multilingua
languages = {
    "it": {
        "window_title": "Progetto Negrisolo R.I.C.O",
        "start_detection": "Avvia Riconoscimento Webcam",
        "screen_detection": "Avvia Riconoscimento Schermo",
        "stop": "Ferma",
        "what_do_you_see": "Cosa vedi?",
        "object_use": "Uso dell'oggetto",
        "detected_objects": "Oggetti rilevati",
        "no_objects_detected": "Nessun oggetto rilevato",
        "save_image": "Salva Immagine",
        "statistics": "Analisi Statistica",
        "night_mode": "Modalità Notte",
        "close_app": "Chiudi Applicazione",
        "chat_log": "Chat Log"
    },
    "en": {
        "window_title": "Project Negrisolo R.I.C.O",
        "start_detection": "Start Webcam Detection",
        "screen_detection": "Start Screen Detection",
        "stop": "Stop",
        "what_do_you_see": "What do you see?",
        "object_use": "Object Use",
        "detected_objects": "Detected Objects",
        "no_objects_detected": "No objects detected",
        "save_image": "Save Image",
        "statistics": "Statistics Analysis",
        "night_mode": "Night Mode",
        "close_app": "Close Application",
        "chat_log": "Chat Log"
    }
}

@lru_cache(maxsize=128)
def get_object_use(obj_name):
    return object_uses.get(obj_name, "Non so a cosa serve questo oggetto.")

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, width=200, height=40, corner_radius=10, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=kwargs.get('bg', '#2d2d2d'), highlightthickness=0)
        self.command = command
        self.text = text
        
        # Colori
        self.normal_color = kwargs.get('button_color', '#007acc')
        self.hover_color = kwargs.get('hover_color', '#0098ff')
        self.text_color = kwargs.get('text_color', '#ffffff')
        
        # Disegna il pulsante
        self.create_rounded_rect(width, height, corner_radius)
        self.create_text(width/2, height/2, text=text, 
                        fill=self.text_color, font=('Roboto', 11))
        
        # Binding eventi
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
        
    def create_rounded_rect(self, width, height, radius):
        self.normal_rect = self.create_rounded_rectangle(2, 2, width-2, height-2, 
                                                       radius=radius, fill=self.normal_color)
    
    def create_rounded_rectangle(self, x1, y1, x2, y2, radius, **kwargs):
        points = [x1+radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def on_enter(self, event):
        self.itemconfig(self.normal_rect, fill=self.hover_color)
    
    def on_leave(self, event):
        self.itemconfig(self.normal_rect, fill=self.normal_color)
    
    def on_click(self, event):
        if self.command:
            self.command()

class ObjectDetectionApp:
    def __init__(self, root, lang="it"):
        self.root = root
        self.lang = lang
        self.minimized = False
        self.root.title("Progetto Negrisolo R.I.C.O")
        
        # Rimuovi la barra del titolo
        self.root.overrideredirect(True)
        
        # Imposta dimensione fissa della finestra
        window_width = 1920
        window_height = 1080
        
        # Centra la finestra
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Definizione dei colori del tema
        self.colors = {
            'primary': "#1a1a1a",      # Sfondo principale
            'secondary': "#2d2d2d",    # Sfondo secondario
            'accent': "#007acc",       # Colore di accento
            'text': "#ffffff",         # Colore del testo
            'error': "#dc3545",        # Colore per errori/warning
            'success': "#28a745",      # Colore per successo
            'warning': "#ffc107"       # Colore per warning
        }

        # Font dell'applicazione
        self.fonts = {
            'title': ("Roboto", 24, "bold"),
            'subtitle': ("Roboto", 18),
            'body': ("Roboto", 12),
            'small': ("Roboto", 10)
        }

        # Inizializza l'interfaccia
        self.setup_title_bar()
        self.setup_menu()
        self.setup_frames()
        self.setup_buttons()
        self.setup_chat()

        self.night_mode = False  # Aggiungi attributo per la modalità notte

        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.screen_running = False
        self.detected_objects = []
        self.model = None  # Defer model loading
        self.history = []  # Storico degli oggetti rilevati

        self.minimized = False  # Aggiungi questa variabile

        self.recording = False
        self.video_recorder = VideoRecorder()

        # Inizializza il database
        self.init_db()

        # Carica le immagini per il pulsante modalità notte
        self.load_night_mode_images()

        # Crea il file di segnalazione per indicare che l'app è completamente caricata
        with open("app_loaded.signal", "w") as f:
            f.write("loaded")

        # Configurazione delle prestazioni
        cv2.setNumThreads(4)  # Limita il numero di thread OpenCV
        self.root.update_idletasks()  # Forza l'aggiornamento dell'interfaccia
        self.root.after(100, self._init_delayed)  # Inizializzazione ritardata

        self._setup_resource_management()
        self._cache = {}  # Cache per i risultati
        self._last_detection_time = 0
        self._detection_interval = 0.1  # Intervallo minimo tra rilevamenti (secondi)

        # Aggiungi questo dopo aver impostato overrideredirect
        self.root.bind('<Map>', self._on_deiconify)
        self.root.bind('<Unmap>', lambda e: setattr(self, 'minimized', True))

    def _setup_resource_management(self):
        # Pulisci le risorse quando l'app viene chiusa
        self.root.protocol("WM_DELETE_WINDOW", self._cleanup)
        
        # Limita l'uso della CPU
        self._max_fps = 30
        self._frame_time = 1.0 / self._max_fps

    def _cleanup(self):
        self.stop_detection()
        self.conn.close()
        cv2.destroyAllWindows()
        self.root.quit()

    def _init_delayed(self):
        # Carica il modello in background
        threading.Thread(target=self.load_model, daemon=True).start()

    def hide(self):
        self.root.withdraw()

    def show(self):
        self.root.deiconify()
        self.root.overrideredirect(True)
        # Ripristina dimensione normale invece dello schermo intero
        window_width = 1280
        window_height = 720
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.minimized = False

    def setup_menu(self):
        # Creazione della barra dei menu
        self.menubar = Menu(self.root)
        self.root.config(menu=self.menubar)

        # Menu File
        self.file_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Salva Immagine", command=self.save_image)
        self.file_menu.add_command(label="Esporta Statistiche", command=self.export_statistics)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Esci", command=self.root.quit)

        # Menu Strumenti
        self.tools_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Strumenti", menu=self.tools_menu)
        self.tools_menu.add_command(label="Analisi Statistica", command=self.show_statistics)
        self.tools_menu.add_command(label="Pulisci Chat", command=self.clear_chat)

        # Menu Lingua
        self.language_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Lingua", menu=self.language_menu)
        self.language_menu.add_command(label="Italiano", command=lambda: self.change_language("it"))
        self.language_menu.add_command(label="English", command=lambda: self.change_language("en"))

        # Menu Video
        self.video_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Video", menu=self.video_menu)
        self.video_menu.add_command(label="Schermo Intero", command=self.toggle_fullscreen)

    def toggle_fullscreen(self):
        self.fullscreen = not getattr(self, 'fullscreen', False)
        self.root.attributes("-fullscreen", self.fullscreen)
        if not self.fullscreen:
            self.root.geometry(f"{1920}x{1080}")

    def setup_styles(self):
        # Stile personalizzato
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Comfortaa", 12), padding=10, relief="flat", borderwidth=0)
        self.style.map("TButton", 
                       background=[('active', '#3E4149')],
                       relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        # Stile per il pulsante modalità notte
        self.style.configure("NightMode.TButton", background="#2C2F33", borderwidth=0)
        self.style.map("NightMode.TButton", background=[('active', '#2C2F33')])

    def setup_frames(self):
        # Frame principale con griglia responsiva
        self.main_frame = Frame(self.root, bg=self.colors['primary'])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.grid_columnconfigure(1, weight=3)  # Dai più peso alla colonna del video
        self.main_frame.grid_columnconfigure(2, weight=1)  # Peso minore per la chat
        
        # Sidebar per i controlli
        self.sidebar = Frame(self.main_frame, bg=self.colors['secondary'], width=200)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Area principale con video (centrata e più grande)
        self.content_frame = Frame(self.main_frame, bg=self.colors['primary'])
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.content_frame.grid_propagate(False)  # Impedisce il ridimensionamento automatico
        
        # Video frame con bordo arrotondato
        self.video_frame = Frame(
            self.content_frame,
            bg=self.colors['secondary'],
            highlightthickness=2,
            highlightbackground=self.colors['accent']
        )
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Area dedicata al video (più grande)
        self.label = Label(self.video_frame, bg=self.colors['secondary'])
        self.label.pack(fill=tk.BOTH, expand=True)

        # Label per oggetti rilevati sotto il video
        self.detected_label = Label(
            self.content_frame, 
            text="", 
            font=self.fonts['subtitle'], 
            bg=self.colors['primary'], 
            fg=self.colors['text']
        )
        self.detected_label.pack(fill=tk.X, padx=10, pady=5)

        # Chat frame moderno (più stretto)
        self.chat_frame = Frame(self.main_frame, bg=self.colors['secondary'], width=300)
        self.chat_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        
        # Titolo chat
        Label(
            self.chat_frame,
            text=languages[self.lang]["chat_log"],
            font=self.fonts['subtitle'],
            bg=self.colors['secondary'],
            fg=self.colors['text']
        ).pack(pady=10)
        
        # Area chat con bordo arrotondato
        self.chat_text = Text(
            self.chat_frame,
            wrap=tk.WORD,
            bg=self.colors['primary'],
            fg=self.colors['text'],
            font=self.fonts['body'],
            relief='flat',
            padx=10,
            pady=10,
            height=20  # Altezza ridotta
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar moderna
        self.scrollbar = ttk.Scrollbar(self.chat_frame, orient="vertical")
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_text.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.configure(command=self.chat_text.yview)

    def setup_buttons(self):
        # Container per i pulsanti con spaziatura uniforme
        self.button_container = Frame(self.sidebar, bg=self.colors['secondary'])
        self.button_container.pack(fill=tk.X, pady=20, padx=10)
        
        buttons = [
            (languages[self.lang]["start_detection"], self.start_detection),
            (languages[self.lang]["screen_detection"], self.toggle_screen_detection),
            (languages[self.lang]["stop"], self.stop_detection),
            (languages[self.lang]["what_do_you_see"], self.show_detected_objects),
            (languages[self.lang]["object_use"], self.show_object_use),
            (languages[self.lang]["save_image"], self.save_image),
            (languages[self.lang]["statistics"], self.show_statistics)
        ]
        
        # Aumenta la larghezza dei pulsanti
        button_width = 250  # Aumentato da 180
        
        for text, command in buttons:
            btn = RoundedButton(
                self.button_container,
                text=text,
                command=command,
                width=button_width,  # Larghezza aumentata
                height=40,          # Altezza aumentata da 35
                corner_radius=10,
                button_color=self.colors['accent'],
                hover_color=self.colors['secondary'],
                text_color=self.colors['text']
            )
            btn.pack(pady=5, padx=5)

        # Pulsante modalità notte con stile arrotondato
        self.night_mode_button = RoundedButton(
            self.button_container,
            text=languages[self.lang]["night_mode"],
            command=self.toggle_night_mode,
            width=button_width,  # Larghezza aumentata
            height=40,          # Altezza aumentata
            corner_radius=10,
            button_color='#2d2d2d',
            hover_color='#3d3d3d'
        )
        self.night_mode_button.pack(pady=5, padx=5)

        # Pulsante chiudi con stile arrotondato
        self.close_button = RoundedButton(
            self.button_container,
            text=languages[self.lang]["close_app"],
            command=self.root.quit,
            width=button_width,  # Larghezza aumentata
            height=40,          # Altezza aumentata
            corner_radius=10,
            button_color=self.colors['error'],
            hover_color='#ff3545'
        )
        self.close_button.pack(pady=5, padx=5)

        advanced_buttons = [
            ("🎥 Registra Video", self.toggle_recording),
            ("🎤 Controllo Vocale", self.toggle_voice_control),
            ("📊 Statistiche Avanzate", self.show_advanced_stats),
            ("⚠️ Gestione Allerte", self.manage_alerts),
            ("📄 Genera Report", self.generate_pdf_report)
        ]
        
        for text, command in advanced_buttons:
            btn = RoundedButton(
                self.button_container,
                text=text,
                command=command,
                width=button_width,
                height=40,
                corner_radius=10,
                button_color=self.colors['accent'],
                hover_color=self.colors['secondary']
            )
            btn.pack(pady=5, padx=5)

    def setup_chat(self):
        # Chat frame moderno
        self.chat_frame = Frame(self.main_frame, bg=self.colors['secondary'], width=300)
        self.chat_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        
        # Titolo chat
        Label(
            self.chat_frame,
            text=languages[self.lang]["chat_log"],
            font=self.fonts['subtitle'],
            bg=self.colors['secondary'],
            fg=self.colors['text']
        ).pack(pady=10)
        
        # Area chat con bordo arrotondato
        self.chat_text = Text(
            self.chat_frame,
            wrap=tk.WORD,
            bg=self.colors['primary'],
            fg=self.colors['text'],
            font=self.fonts['body'],
            relief='flat',
            padx=10,
            pady=10
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar moderna
        self.scrollbar = ttk.Scrollbar(self.chat_frame, orient="vertical")
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_text.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.configure(command=self.chat_text.yview)

    def setup_title_bar(self):
        # Barra del titolo personalizzata
        self.title_bar = Frame(self.root, bg=self.colors['secondary'], height=30)
        self.title_bar.pack(fill=tk.X)
        
        # Titolo
        title_label = Label(
            self.title_bar, 
            text="R.I.C.O",
            font=self.fonts['subtitle'],
            bg=self.colors['secondary'],
            fg=self.colors['text']
        )
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Pulsanti di controllo
        minimize_btn = RoundedButton(
            self.title_bar,
            text="−",
            command=self.minimize_window,  # Cambia qui
            width=40,
            height=25,
            corner_radius=5,
            button_color=self.colors['secondary'],
            hover_color=self.colors['accent']
        )
        minimize_btn.pack(side=tk.RIGHT, padx=5)

        # Binding per il trascinamento della finestra
        self.title_bar.bind('<Button-1>', self.start_move)
        self.title_bar.bind('<B1-Motion>', self.do_move)

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def minimize_window(self):
        if not self.minimized:
            # Salva l'ultima posizione e dimensione
            self.last_geometry = self.root.geometry()
            # Disabilita overrideredirect temporaneamente
            self.root.overrideredirect(False)
            # Minimizza la finestra
            self.root.iconify()
            self.minimized = True
            
            # Binding per il ripristino
            self.root.bind('<Map>', self._on_deiconify)

    def _on_deiconify(self, event):
        if self.minimized:
            # Rimuovi il binding
            self.root.unbind('<Map>')
            # Piccolo delay per permettere alla finestra di essere visualizzata
            self.root.after(10, self._restore_window)

    def _restore_window(self):
        # Ripristina overrideredirect
        self.root.overrideredirect(True)
        # Ripristina la dimensione originale invece dello schermo intero
        window_width = 1280
        window_height = 720
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.minimized = False

    def init_db(self):
        self.conn = sqlite3.connect('object_detection.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                               (timestamp TEXT, object TEXT)''')
        self.conn.commit()

    def load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")

    def start_detection(self):
        self.load_model()
        self.running = True
        threading.Thread(target=self.detect_objects, daemon=True).start()
    
    def toggle_screen_detection(self):
        self.load_model()
        self.screen_running = not self.screen_running
        if self.screen_running:
            threading.Thread(target=self.detect_screen_objects, daemon=True).start()
    
    def stop_detection(self):
        self.running = False
        self.screen_running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    
    def detect_objects(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)
            time.sleep(0.01)
    
    def detect_screen_objects(self):
        while self.screen_running:
            try:
                # Usa mss invece di pyautogui per screenshot più veloci
                with mss.mss() as sct:
                    screenshot = sct.grab(sct.monitors[0])
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.process_frame(frame)
                    
                # Rilascia la memoria
                del frame
                gc.collect()
                
                time.sleep(0.1)  # Aumenta l'intervallo tra gli screenshot
                
            except Exception as e:
                logging.error(f"Errore durante la cattura dello schermo: {e}")
                time.sleep(1)  # Pausa più lunga in caso di errore
    
    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self._last_detection_time < self._detection_interval:
            return
            
        try:
            # Resize frame for faster processing
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
            results = self.model(frame)[0]
            detected_objects_temp = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results.names[int(box.cls[0])]
                conf = box.conf[0].item()
                detected_objects_temp.append(label)
                
                # Disegna una sfera bianca dietro il testo
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{label} {conf:.2f}"
                text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
                text_x, text_y = x1, y1 - 10
                radius = max(text_size) // 2 + 5
                cv2.circle(frame, (text_x + text_size[0] // 2, text_y - text_size[1] // 2), radius, (255, 255, 255), -1)
                cv2.putText(frame, text, (text_x, text_y), font, 0.5, (0, 0, 0), 2)
            
            self.detected_objects = detected_objects_temp
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            
            if self.detected_objects:
                objects_seen = ", ".join(set(self.detected_objects))
                self.detected_label.config(text=f"{languages[self.lang]['detected_objects']}: {objects_seen}", fg="#00FF00")
                self.save_to_db(self.detected_objects)
            else:
                self.detected_label.config(text=languages[self.lang]['no_objects_detected'], fg="#FF0000")
        except Exception as e:
            logging.error(f"Errore durante l'elaborazione del frame: {e}")
        
        # Registra il frame se la registrazione è attiva
        if self.recording:
            self.video_recorder.record_frame(frame)
        
        self._last_detection_time = current_time
    
    def save_to_db(self, objects):
        if not objects:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Usa executemany per inserimenti multipli
        self.cursor.executemany(
            "INSERT INTO detections (timestamp, object) VALUES (?, ?)",
            [(timestamp, obj) for obj in set(objects)]
        )
        self.conn.commit()

    def show_detected_objects(self):
        if self.detected_objects:
            objects_seen = ", ".join(set(self.detected_objects))
            response = f"Vedo: {objects_seen}"
        else:
            response = "Non vedo nessun oggetto chiaramente."
        
        self.update_chat(response)

    def show_object_use(self):
        if self.detected_objects:
            obj = self.detected_objects[0]  # Assume the first detected object
            response = get_object_use(obj)
        else:
            response = "Non ho rilevato l'oggetto di cui chiedi l'uso."
        
        self.update_chat(response)

    def save_image(self):
        if self.label.imgtk:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                self.label.imgtk._PhotoImage__photo.write(file_path, format="png")

    def show_statistics(self):
        self.cursor.execute("SELECT object, COUNT(*) FROM detections GROUP BY object")
        stats = self.cursor.fetchall()
        if stats:
            stats_message = "Statistiche degli oggetti rilevati:\n"
            for obj, count in stats:
                stats_message += f"{obj}: {count}\n"
        else:
            stats_message = "Nessun dato statistico disponibile."
        self.update_chat(stats_message)

    def export_statistics(self):
        """Esporta le statistiche in un file CSV."""
        try:
            self.cursor.execute("SELECT object, COUNT(*) FROM detections GROUP BY object")
            stats = self.cursor.fetchall()
            if stats:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                if file_path:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Oggetto', 'Conteggio'])
                        writer.writerows(stats)
                    self.update_chat(f"Statistiche esportate in {file_path}")
            else:
                messagebox.showinfo("Info", "Nessun dato statistico disponibile da esportare.")
        except Exception as e:
            logging.error(f"Errore durante l'esportazione delle statistiche: {e}")
            messagebox.showerror("Errore", "Errore durante l'esportazione delle statistiche.")

    def clear_chat(self):
        """Pulisce la chat."""
        self.chat_text.delete(1.0, tk.END)

    def update_chat(self, message):
        # Limita la dimensione della chat
        MAX_LINES = 1000
        current_lines = self.chat_text.get("1.0", tk.END).count('\n')
        
        if current_lines > MAX_LINES:
            # Rimuovi le prime linee quando supera il limite
            self.chat_text.delete("1.0", f"{current_lines - MAX_LINES}.0")
        
        self.chat_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.chat_text.see(tk.END)

    def load_night_mode_images(self):
        """Carica le immagini per il pulsante modalità notte."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            sun_path = os.path.join(script_dir, "assets", "sun.png")
            moon_path = os.path.join(script_dir, "assets", "moon.png")
            self.sun_img = ImageTk.PhotoImage(Image.open(sun_path).resize((24, 24)))
            self.moon_img = ImageTk.PhotoImage(Image.open(moon_path).resize((24, 24)))
        except Exception as e:
            logging.error(f"Errore nel caricamento delle icone: {e}")
            self.sun_img = None
            self.moon_img = None

    def toggle_night_mode(self):
        """Attiva/disattiva la modalità notte."""
        self.night_mode = not self.night_mode
        if self.night_mode:
            self.root.configure(bg="#000000")
            self.main_frame.configure(bg="#000000")
            self.video_frame.configure(bg="#000000")
            self.button_frame.configure(bg="#000000")
            self.chat_frame.configure(bg="#000000")
            self.detected_label.configure(bg="#000000", fg="#FFFFFF")
            self.chat_text.configure(bg="#000000", fg="#FFFFFF")
        else:
            self.root.configure(bg="#2C2F33")
            self.main_frame.configure(bg="#2C2F33")
            self.video_frame.configure(bg="#23272A")
            self.button_frame.configure(bg="#2C2F33")
            self.chat_frame.configure(bg="#2C2F33")
            self.detected_label.configure(bg="#2C2F33", fg="#FFFFFF")
            self.chat_text.configure(bg="#2C2F33", fg="#FFFFFF")
        self.update_night_mode_button()

    def update_night_mode_button(self):
        """Aggiorna l'icona del pulsante modalità notte."""
        if self.night_mode:
            self.night_mode_button.config(image=self.sun_img, text="")
        else:
            self.night_mode_button.config(image=self.moon_img, text="")

    def toggle_recording(self):
        """Attiva/disattiva la registrazione video"""
        if not self.recording:
            self.video_recorder.start_recording()
            self.recording = True
            self.update_chat("Registrazione video avviata")
        else:
            self.video_recorder.stop_recording()
            self.recording = False
            self.update_chat("Registrazione video terminata")
            
    def toggle_voice_control(self):
        """Attiva/disattiva il controllo vocale"""
        if not hasattr(self, 'voice_assistant'):
            self.voice_assistant = VoiceAssistant()
            self.voice_control_active = True
            threading.Thread(target=self._voice_control_loop, daemon=True).start()
            self.update_chat("Controllo vocale attivato")
        else:
            self.voice_control_active = False
            self.update_chat("Controllo vocale disattivato")
            
    def show_advanced_stats(self):
        """Mostra le statistiche avanzate"""
        try:
            stats = AdvancedStats(self.conn)
            fig = stats.generate_daily_report()
            fig.show()
        except Exception as e:
            self.update_chat(f"Errore durante la generazione delle statistiche: {e}")
            
    def manage_alerts(self):
        """Gestisce le impostazioni delle allerte"""
        if not hasattr(self, 'alert_system'):
            self.alert_system = AlertSystem()
            self.update_chat("Sistema di allerta attivato")
        else:
            self.update_chat("Sistema di allerta già attivo")
            
    def generate_pdf_report(self):
        """Genera un report PDF"""
        try:
            data = {
                'statistics': self._get_detection_stats(),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            report_gen = ReportGenerator()
            filename = f"reports/rico_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_gen.generate_report(data, filename)
            self.update_chat(f"Report generato: {filename}")
        except Exception as e:
            self.update_chat(f"Errore durante la generazione del report: {e}")
            
    def _voice_control_loop(self):
        """Loop per il controllo vocale"""
        while self.voice_control_active:
            command = self.voice_assistant.listen()
            if command:
                self.update_chat(f"Comando vocale riconosciuto: {command}")
                
    def _get_detection_stats(self):
        """Recupera le statistiche di rilevamento"""
        self.cursor.execute("""
            SELECT object, COUNT(*) as count 
            FROM detections 
            GROUP BY object
            ORDER BY count DESC
        """)
        return dict(self.cursor.fetchall())

class VideoRecorder:
    def __init__(self):
        self.recording = False
        self.output = None
        self.filename = None
        
    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"recordings/rico_recording_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output = cv2.VideoWriter(self.filename, fourcc, 20.0, (640,480))
        self.recording = True
        
    def record_frame(self, frame):
        if self.recording and self.output:
            self.output.write(frame)
            
    def stop_recording(self):
        if self.output:
            self.output.release()
            self.recording = False

class VoiceAssistant:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.commands = {
            "avvia riconoscimento": "start_detection",
            "ferma riconoscimento": "stop_detection",
            "cosa vedi": "show_detected_objects",
            "modalità notte": "toggle_night_mode"
        }
        
    def listen(self):
        with sr.Microphone() as source:
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio, language='it-IT')
                return self.process_command(text)
            except sr.UnknownValueError:
                return "Non ho capito il comando"

class AdvancedStats:
    def __init__(self, db_connection):
        self.conn = db_connection
        
    def generate_daily_report(self):
        query = """
        SELECT 
            date(timestamp) as date,
            object,
            COUNT(*) as count
        FROM detections
        GROUP BY date(timestamp), object
        ORDER BY date
        """
        df = pd.read_sql_query(query, self.conn)
        
        fig = px.line(df, x='date', y='count', color='object',
                     title='Rilevamenti Giornalieri per Oggetto')
        return fig

class AlertSystem:
    def __init__(self):
        self.alert_objects = {
            "person": "Persona rilevata!",
            "knife": "⚠️ Oggetto pericoloso rilevato!",
            "fire": "🔥 Incendio rilevato!"
        }
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("assets/alert.wav")
        
    def check_alerts(self, detected_objects):
        alerts = []
        for obj in detected_objects:
            if (obj in self.alert_objects):
                alerts.append({
                    "time": datetime.now().isoformat(),
                    "object": obj,
                    "message": self.alert_objects[obj]
                })
                self.alert_sound.play()
        return alerts

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
    def generate_report(self, data, output_file):
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        story = []
        
        # Aggiungi titolo
        story.append(Paragraph("Report Rilevamenti RICO", self.styles['Title']))
        story.append(Spacer(1, 12))
        
        # Aggiungi statistiche
        for obj, count in data['statistics'].items():
            text = f"{obj}: {count} rilevamenti"
            story.append(Paragraph(text, self.styles['Normal']))

class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("R.I.C.O")
        
        # Rimuovi la barra del titolo di default
        self.root.overrideredirect(True)
        
        # Ottieni dimensioni dello schermo
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Imposta dimensioni e posizione
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Colori moderni
        self.colors = {
            'bg': "#0A1929",         # Blu scuro
            'accent': "#007FFF",     # Blu chiaro
            'text': "#FFFFFF",       # Bianco
            'secondary': "#132F4C"   # Blu medio
        }
        
        # Configurazione finestra
        self.root.configure(bg=self.colors['bg'])
        self.root.attributes('-topmost', True)
        
        # Frame principale centrato
        self.main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Logo animato
        self.logo = tk.Label(
            self.main_frame,
            text="R.I.C.O",
            font=("Montserrat", 72, "bold"),
            fg=self.colors['accent'],
            bg=self.colors['bg']
        )
        self.logo.pack()
        
        # Sottotitolo
        self.subtitle = tk.Label(
            self.main_frame,
            text="Ricerca Intermittente Centro Oggetti",
            font=("Montserrat", 24),
            fg=self.colors['text'],
            bg=self.colors['bg']
        )
        self.subtitle.pack(pady=(0, 40))
        
        # Progress bar moderna
        style = ttk.Style()
        style.configure(
            "Modern.Horizontal.TProgressbar",
            thickness=4,
            troughcolor=self.colors['secondary'],
            background=self.colors['accent']
        )
        
        self.progress = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            style="Modern.Horizontal.TProgressbar",
            length=400,
            mode='determinate',
            variable=self.progress
        )
        self.progress_bar.pack()
        
        # Label per lo stato del caricamento
        self.status_label = tk.Label(
            self.main_frame,
            text="Caricamento...",
            font=("Montserrat", 12),
            fg=self.colors['text'],
            bg=self.colors['bg']
        )
        self.status_label.pack(pady=20)
        
        # Versione in basso a destra
        self.version_label = tk.Label(
            self.main_frame,
            text="V2.0",
            font=("Montserrat", 16, "bold"),
            fg="#FF0000",
            bg=self.colors['bg']
        )
        self.version_label.pack(side=tk.RIGHT, anchor=tk.SE, padx=20, pady=20)
        
        # Avvia animazione e caricamento
        self.progress_value = 0
        self.start_time = time.time()
        self.animate_logo()
        self.update_progress()
        
    def animate_logo(self):
        def pulse(step=0):
            scale = 1.0 + 0.05 * math.sin(step)
            self.logo.configure(font=("Montserrat", int(72 * scale), "bold"))
            self.root.after(50, pulse, step + 0.2)
        pulse()
        
    def update_progress(self):
        elapsed_time = time.time() - self.start_time
        progress = min(100, (elapsed_time / 15.0) * 100)  # 15 secondi durata
        
        self.progress_value = progress
        self.progress.set(self.progress_value)
        
        if progress < 33:
            self.status_label.config(text="Inizializzazione...")
        elif progress < 66:
            self.status_label.config(text="Caricamento moduli...")
        else:
            self.status_label.config(text="Completamento...")
        
        if progress < 100:
            self.root.after(50, self.update_progress)
        else:
            self.root.destroy()

if __name__ == "__main__":
    # Mostra splash screen
    splash_root = tk.Tk()
    splash = SplashScreen(splash_root)
    splash_root.mainloop()
    
    # Avvia l'applicazione principale
    root = tk.Tk()
    app = ObjectDetectionApp(root, lang="it")
    root.mainloop()
