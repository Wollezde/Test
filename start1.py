# start.py

import tkinter as tk
from tkinter import Menu, ttk, messagebox
import configparser
import threading
import os
from cam_setting import CameraAndServoControlApp  # Stelle sicher, dass dieses Modul verfügbar ist
from vidgear.gears import VideoGear
import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
from PIL import Image, ImageTk
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


class StartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Start Menu")
        self.root.state('zoomed')
        self.config = configparser.ConfigParser()
        self.config_file = "config.ini"
        self.ser = None  # Für die serielle Verbindung mit Arduino

        # Grid-Konfiguration für das Hauptfenster
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)

        # Gesichtserkennungs-Klassifikator laden
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Gesichtserkennungsmodell initialisieren
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            self.recognizer = cv2.face_LBPHFaceRecognizer.create()

        self.face_data_folder = "Gesichtserkennung"
        if not os.path.exists(self.face_data_folder):
            os.makedirs(self.face_data_folder)

        # Gesichter und Labels laden
        self.face_labels = {}
        self.trained = False  # Initialisierungsstatus des Modells

        # Konfigurationsdatei laden
        self.load_config()

        # Anzahl der Bilder pro Person aus der Konfiguration laden
        self.max_samples = self.config.getint(
            "FaceRecognition", "max_samples", fallback=30)

        # Thread zur Anzeige der Kamerabilder
        self.stop_event = threading.Event()

        # Variablen für die Gesichtsdatenaufnahme
        self.collecting_faces = False
        self.face_samples = []
        self.samples_collected = 0

        # Variablen für unbekannte Gesichter
        self.unknown_faces = {}
        self.face_counter = 1  # Zähler für unbekannte Gesichter

        # GUI erstellen
        self.create_menu()
        self.create_controls()

        # Port-Verbindung herstellen
        self.connect_to_arduino()

        # Lade vorhandene Gesichter
        self.load_known_faces()

        # Kamera-Streams starten
        self.update_camera_sources()
        self.camera_thread = threading.Thread(
            target=self.update_camera_frames, daemon=True)
        self.camera_thread.start()

        logging.info("StartApp initialisiert.")

    def load_config(self):
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            # Falls die Konfigurationsdatei nicht existiert, erstelle sie und füge Standardwerte hinzu
            self.config["FaceRecognition"] = {"max_samples": "30"}
            self.save_config()

    def save_config(self):
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    def create_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # Datei-Menü
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(label="Beenden", command=self.exit_app)

        # Konfigurations-Menü
        config_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Konfiguration", menu=config_menu)
        config_menu.add_command(
            label="Kamera", command=self.open_camera_settings)
        config_menu.add_command(label="Port", command=self.open_port_settings)
        config_menu.add_command(
            label="Gesichtserkennung", command=self.open_face_recognition_settings)

    def create_controls(self):
        # Frame für die Kameravorschau
        self.camera_frame = tk.Frame(self.root)
        self.camera_frame.grid(
            row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.camera_frame.grid_propagate(False)
        self.camera_frame.columnconfigure(0, weight=1)
        self.camera_frame.columnconfigure(1, weight=1)
        self.camera_frame.rowconfigure(0, weight=1)

        # Label für die linke Kamera
        self.camera_label_1 = tk.Label(self.camera_frame, bg="black")
        self.camera_label_1.grid(
            row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Label für die rechte Kamera
        self.camera_label_2 = tk.Label(self.camera_frame, bg="black")
        self.camera_label_2.grid(
            row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Button zum Aufnehmen eines Einzelbildes
        self.capture_button = tk.Button(
            self.root, text="Bild aufnehmen", command=self.capture_faces)
        self.capture_button.grid(row=1, column=0, pady=5)

        # Frame für die Steuerung der Augenbewegungen
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(
            row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Steuerungsbuttons
        self.up_button = tk.Button(
            self.control_frame, text="Up", command=self.move_up, width=10)
        self.up_button.grid(row=0, column=1, pady=5)

        self.left_button = tk.Button(
            self.control_frame, text="Left", command=self.move_left, width=10)
        self.left_button.grid(row=1, column=0, padx=5)

        self.center_button = tk.Button(
            self.control_frame, text="Go to Center", command=self.move_to_center_all, width=10)
        self.center_button.grid(row=1, column=1, padx=5)

        self.right_button = tk.Button(
            self.control_frame, text="Right", command=self.move_right, width=10)
        self.right_button.grid(row=1, column=2, padx=5)

        self.down_button = tk.Button(
            self.control_frame, text="Down", command=self.move_down, width=10)
        self.down_button.grid(row=2, column=1, pady=5)

    def open_camera_settings(self):
        # Kamera-Streams stoppen, bevor cam_setting geöffnet wird
        self.stop_camera_streams()

        cam_root = tk.Toplevel(self.root)
        CameraAndServoControlApp(cam_root, existing_serial=self.ser)

        # Kamera-Streams neu starten, nachdem die Einstellungen geschlossen wurden
        cam_root.protocol("WM_DELETE_WINDOW", lambda: [
                          cam_root.destroy(), self.restart_camera_streams()])

    def open_port_settings(self):
        port_root = tk.Toplevel(self.root)
        port_root.title("Port Configuration")
        port_root.geometry("300x150")

        available_ports = [
            port.device for port in serial.tools.list_ports.comports()]
        selected_port = tk.StringVar(
            value=self.config.get("PortConfig", "selected_port", fallback=""))

        tk.Label(port_root, text="Select Serial Port:").pack(pady=10)
        port_combobox = ttk.Combobox(
            port_root, textvariable=selected_port, values=available_ports, state="readonly")
        port_combobox.pack(pady=5)

        def save_port():
            if not self.config.has_section("PortConfig"):
                self.config.add_section("PortConfig")
            self.config.set("PortConfig", "selected_port",
                            selected_port.get())
            self.save_config()
            messagebox.showinfo("Info", "Port configuration saved.")
            port_root.destroy()
            # Verbindung zum Arduino neu herstellen
            self.connect_to_arduino()

        save_button = tk.Button(port_root, text="Save", command=save_port)
        save_button.pack(pady=10)

    def open_face_recognition_settings(self):
        face_root = tk.Toplevel(self.root)
        face_root.title("Gesichtserkennung Einstellungen")
        face_root.geometry("300x150")

        max_samples_var = tk.IntVar(value=self.max_samples)

        tk.Label(face_root, text="Anzahl der Bilder pro Person:").pack(pady=10)
        max_samples_entry = tk.Entry(
            face_root, textvariable=max_samples_var)
        max_samples_entry.pack(pady=5)

        def save_face_settings():
            self.max_samples = max_samples_var.get()
            if not self.config.has_section("FaceRecognition"):
                self.config.add_section("FaceRecognition")
            self.config.set("FaceRecognition", "max_samples",
                            str(self.max_samples))
            self.save_config()
            messagebox.showinfo(
                "Info", "Gesichtserkennungseinstellungen gespeichert.")
            face_root.destroy()

        save_button = tk.Button(
            face_root, text="Save", command=save_face_settings)
        save_button.pack(pady=10)

    def stop_camera_streams(self):
        if hasattr(self, 'stream1'):
            self.stream1.stop()
        if hasattr(self, 'stream2'):
            self.stream2.stop()
        self.stop_event.set()

    def restart_camera_streams(self):
        # Beende den aktuellen Thread und starte die Kamera-Streams neu
        self.stop_event.set()
        if hasattr(self, 'stream1'):
            self.stream1.stop()
        if hasattr(self, 'stream2'):
            self.stream2.stop()

        # Neue Event- und Thread-Objekte erstellen
        self.stop_event = threading.Event()
        self.update_camera_sources()
        self.camera_thread = threading.Thread(
            target=self.update_camera_frames, daemon=True)
        self.camera_thread.start()

    def connect_to_arduino(self):
        port = self.config.get("PortConfig", "selected_port", fallback=None)
        if port:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = serial.Serial(port, 115200, timeout=1)
                time.sleep(2)  # Warte auf die Verbindung
                messagebox.showinfo(
                    "Info", f"Connected to Arduino on port {port}")
            except serial.SerialException as e:
                messagebox.showerror(
                    "Error", f"Could not open serial port: {e}")
                self.ser = None

    def update_camera_sources(self):
        # Kameraquellen laden
        camera_source_1 = self.config.get(
            "CameraControl", "camera_source_1", fallback="0")
        camera_source_2 = self.config.get(
            "CameraControl", "camera_source_2", fallback="1")

        self.stream1 = VideoGear(
            source=int(camera_source_1), logging=True).start()
        self.stream2 = VideoGear(
            source=int(camera_source_2), logging=True).start()

    def update_camera_frames(self):
        try:
            while not self.stop_event.is_set():
                frameA = self.stream1.read()
                frameB = self.stream2.read()

                if frameA is None or frameB is None:
                    logging.warning("Kamerastream nicht verfügbar.")
                    break

                # Frames verarbeiten
                frameA_display = self.process_frame(frameA, "A")
                frameB_display = self.process_frame(frameB, "B")

                # Konvertiere Bilder für die Anzeige in Tkinter
                img1 = cv2.resize(frameA_display, (500, 400))
                img1 = Image.fromarray(img1)
                photo1 = ImageTk.PhotoImage(image=img1)

                img2 = cv2.resize(frameB_display, (500, 400))
                img2 = Image.fromarray(img2)
                photo2 = ImageTk.PhotoImage(image=img2)

                # Aktualisiere die Labels im Hauptthread
                self.camera_label_1.after(
                    0, self.update_label_image, self.camera_label_1, photo1)
                self.camera_label_2.after(
                    0, self.update_label_image, self.camera_label_2, photo2)
        except Exception as e:
            logging.exception("Fehler in update_camera_frames: %s", e)
        finally:
            # Kamera-Streams stoppen, wenn die Schleife beendet wird
            self.stream1.stop()
            self.stream2.stop()

    def process_frame(self, frame, camera_id):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            for (x, y, w, h) in faces:
                face_image = gray[y:y+h, x:x+w]
                face_id, confidence = self.recognize_face(face_image)

                if face_id is None:
                    # Unbekanntes Gesicht
                    uid = self.get_unknown_face_id(face_image)
                    label = f"Unbekannt_{uid}"
                    rectangle_color = (0, 0, 255)  # Rot für unbekannte Gesichter
                else:
                    label = self.face_labels.get(
                        face_id, f"Person_{face_id}")
                    rectangle_color = (255, 0, 0)  # Blau für bekannte Gesichter

                # Rechteck und Label anzeigen
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              rectangle_color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )
        except Exception as e:
            logging.exception("Fehler in process_frame: %s", e)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def get_unknown_face_id(self, face_image):
        # Prüfe, ob dieses Gesicht bereits in unknown_faces ist
        for uid, data in self.unknown_faces.items():
            if np.array_equal(data['image'], face_image):
                return uid
        # Neues unbekanntes Gesicht hinzufügen
        uid = self.face_counter
        self.unknown_faces[uid] = {'image': face_image}
        self.face_counter += 1
        return uid

    def capture_faces(self):
        # Nimm ein Einzelbild auf
        frame = self.stream1.read()
        if frame is None:
            messagebox.showerror("Fehler", "Konnte kein Bild aufnehmen.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        if len(faces) == 0:
            messagebox.showinfo("Info", "Keine Gesichter im Bild erkannt.")
            return

        # Unbekannte Gesichter im Bild sammeln
        unknown_faces_in_image = {}
        self.face_counter = 1  # Zähler zurücksetzen

        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            face_id, confidence = self.recognize_face(face_image)

            if face_id is None:
                uid = self.face_counter
                unknown_faces_in_image[uid] = face_image
                self.face_counter += 1

        if not unknown_faces_in_image:
            messagebox.showinfo(
                "Info", "Keine unbekannten Gesichter im Bild.")
            return

        # Fenster zur Namenseingabe öffnen
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Unbekannte Gesichter speichern")

        canvas = tk.Canvas(capture_window)
        scrollbar = tk.Scrollbar(
            capture_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.name_entries = {}

        for uid, face_image in unknown_faces_in_image.items():
            frame = tk.Frame(scrollable_frame)
            frame.pack(pady=5, padx=10)

            tk.Label(frame, text=f"Unbekanntes Gesicht {uid}").pack()

            face_img = cv2.resize(face_image, (200, 200))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(face_img)
            photo = ImageTk.PhotoImage(image=img)

            label = tk.Label(frame, image=photo)
            label.image = photo  # Referenz speichern
            label.pack()

            name_label = tk.Label(frame, text="Name eingeben:")
            name_label.pack()

            name_entry = tk.Entry(frame)
            name_entry.pack()
            self.name_entries[uid] = {
                'entry': name_entry, 'face_image': face_image}

        button_frame = tk.Frame(capture_window)
        button_frame.pack(pady=10)

        ok_button = tk.Button(
            button_frame, text="OK", command=lambda: self.save_captured_faces(capture_window))
        ok_button.pack(side="left", padx=5)

        cancel_button = tk.Button(
            button_frame, text="Abbrechen", command=capture_window.destroy)
        cancel_button.pack(side="left", padx=5)

    def save_captured_faces(self, window):
        try:
            # Erstelle ein Mapping von Namen zu face_ids
            name_to_id = {name: face_id for face_id, name in self.face_labels.items()}

            # Maximal vorhandene face_id ermitteln
            max_face_id = max(self.face_labels.keys(), default=0)

            for uid, data in self.name_entries.items():
                name = data['entry'].get().strip()
                if not name:
                    messagebox.showwarning(
                        "Warnung", f"Bitte einen Namen für Gesicht {uid} eingeben.")
                    return

                face_image = data['face_image']

                if name in name_to_id:
                    face_id = name_to_id[name]
                else:
                    # Neuer Eintrag
                    max_face_id += 1
                    face_id = max_face_id
                    name_to_id[name] = face_id
                    self.face_labels[face_id] = name

                # Gesicht speichern
                person_folder = os.path.join(
                    self.face_data_folder, name)
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)
                    logging.debug(f"Ordner erstellt: {person_folder}")

                # Anzahl der bereits vorhandenen Bilder zählen
                existing_images = [f for f in os.listdir(
                    person_folder) if f.endswith('.png')]
                image_count = len(existing_images)

                face_filename = os.path.join(
                    person_folder, f"{name}_{image_count}.png")
                cv2.imwrite(face_filename, face_image)
                logging.debug(f"Gesichtsbild gespeichert: {face_filename}")

            # Labels speichern
            self.save_face_labels()

            # Modell neu trainieren
            self.train_recognizer()

            # Unbekannte Gesichter zurücksetzen
            self.unknown_faces.clear()
            self.face_counter = 1

            window.destroy()
            logging.info(
                "Unbekannte Gesichter gespeichert und Modell trainiert.")
        except Exception as e:
            logging.exception("Fehler in save_captured_faces: %s", e)

    def recognize_face(self, face_image):
        if not self.trained:
            return None, None

        try:
            label, confidence = self.recognizer.predict(face_image)
            if confidence < 70:
                return label, confidence
            else:
                return None, None
        except Exception as e:
            logging.exception("Fehler in recognize_face: %s", e)
            return None, None

    def save_face_labels(self):
        labels_file = os.path.join(self.face_data_folder, "labels.txt")
        try:
            with open(labels_file, "w", encoding="utf-8") as f:
                for face_id, name in self.face_labels.items():
                    f.write(f"{face_id}:{name}\n")
            logging.debug("Labels gespeichert.")
        except Exception as e:
            logging.exception(
                "Fehler beim Speichern der Labels: %s", e)

    def load_known_faces(self):
        try:
            # Lade vorhandene Gesichter und trainiere das Modell
            self.face_labels = {}
            face_images = []
            face_ids = []

            labels_file = os.path.join(
                self.face_data_folder, "labels.txt")
            if os.path.exists(labels_file):
                with open(labels_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        face_id, name = line.strip().split(":", 1)
                        face_id = int(face_id)
                        self.face_labels[face_id] = name

            for face_id, name in self.face_labels.items():
                person_folder = os.path.join(
                    self.face_data_folder, name)
                if os.path.isdir(person_folder):
                    for filename in os.listdir(person_folder):
                        if filename.endswith(".png"):
                            img_path = os.path.join(
                                person_folder, filename)
                            face_image = cv2.imread(
                                img_path, cv2.IMREAD_GRAYSCALE)
                            if face_image is None:
                                logging.warning(
                                    f"Konnte Bild nicht laden: {img_path}")
                                continue
                            face_images.append(face_image)
                            face_ids.append(face_id)

            if face_images and face_ids:
                self.recognizer.train(face_images, np.array(face_ids))
                self.trained = True
                logging.info(
                    "Gesichtserkennungsmodell erfolgreich trainiert.")
            else:
                self.trained = False
                logging.warning(
                    "Keine Gesichtsbilder zum Trainieren gefunden.")
        except Exception as e:
            logging.exception("Fehler in load_known_faces: %s", e)

    def train_recognizer(self):
        try:
            face_images = []
            face_ids = []

            for face_id, name in self.face_labels.items():
                person_folder = os.path.join(
                    self.face_data_folder, name)
                if os.path.isdir(person_folder):
                    for filename in os.listdir(person_folder):
                        if filename.endswith(".png"):
                            img_path = os.path.join(
                                person_folder, filename)
                            face_image = cv2.imread(
                                img_path, cv2.IMREAD_GRAYSCALE)
                            if face_image is None:
                                logging.warning(
                                    f"Konnte Bild nicht laden: {img_path}")
                                continue
                            face_images.append(face_image)
                            face_ids.append(face_id)

            if face_images and face_ids:
                self.recognizer.train(face_images, np.array(face_ids))
                self.trained = True
                logging.info(
                    "Gesichtserkennungsmodell erfolgreich trainiert.")
            else:
                self.trained = False
                logging.warning(
                    "Keine Gesichtsbilder zum Trainieren gefunden.")
        except Exception as e:
            logging.exception("Fehler in train_recognizer: %s", e)

    def update_label_image(self, label, image):
        label.configure(image=image)
        label.image = image

    def move_eyes(self, direction):
        if self.ser:
            servo_settings = {
                "up": [("Servo_2", "up"), ("Servo_3", "up")],
                "down": [("Servo_2", "down"), ("Servo_3", "down")],
                "left": [("Servo_0", "down"), ("Servo_1", "down")],
                "right": [("Servo_0", "up"), ("Servo_1", "up")],
                "center": [
                    ("Servo_1", "center"),
                    ("Servo_0", "center"),
                    ("Servo_3", "center"),
                    ("Servo_2", "center"),
                ],
            }

            commands = servo_settings.get(direction)
            if commands:
                for servo, action in commands:
                    self.send_servo_command(servo, action)

    def send_servo_command(self, servo, action):
        if not self.config.has_section(servo):
            return

        angle = self.config.getint(servo, action, fallback=90)
        reverse = self.config.getboolean(servo, "reverse", fallback=False)
        port = self.config.getint(servo, "port", fallback=None)

        # Wenn reverse aktiviert ist, die Winkelwerte entsprechend anpassen
        if reverse:
            if action == "up":
                angle = self.config.getint(servo, "down", fallback=90)
            elif action == "down":
                angle = self.config.getint(servo, "up", fallback=90)

        if port is not None:
            command = f"{port}:{angle}\n"
            self.ser.write(command.encode("utf-8"))
            logging.debug(f"Sending command to servo {port}: {angle}")

    def move_up(self):
        self.move_eyes("up")

    def move_down(self):
        self.move_eyes("down")

    def move_left(self):
        self.move_eyes("left")

    def move_right(self):
        self.move_eyes("right")

    def move_to_center_all(self):
        self.move_eyes("center")

    def exit_app(self):
        # Beende den Kamerathread und schließe die Anwendung
        self.stop_event.set()
        if hasattr(self, 'stream1'):
            self.stream1.stop()
        if hasattr(self, 'stream2'):
            self.stream2.stop()
        if self.ser:
            self.ser.close()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = StartApp(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()
