import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from pathlib import Path
import threading
import time
import random

# -----------------------------
# PATH CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# 👉 Path to your trained breast cancer model
# Make sure this file exists:
# C:\Users\AbdulBashar\Downloads\Breast Cancer Detection.v2i.yolov8\YOLOv8\bc_detection\weights\best.pt
MODEL_PATH = BASE_DIR / "YOLOv8" / "bc_detection" / "weights" / "best.pt"

if not MODEL_PATH.exists():
    messagebox.showerror(
        "Model not found",
        f"Could not find model at:\n{MODEL_PATH}\n\nPlease check the path."
    )

# Load YOLO model
model = YOLO(str(MODEL_PATH))

stop_webcam = False
animation_running = False

# -----------------------------
# MATRIX HACKER THEME COLORS
# -----------------------------
BG = "#270b1f"
SIDEBAR_BG = "#74095F"
PANEL_BG = "#801B76"
BTN_BG = "#A71F9E"
BTN_HOVER = "#C420B1"
TEXT = "#e4c0dc"
MUTED = "#f11ece"
ACCENT = "#d70bb5"
ACCENT_SOFT = "#290521"

panel = None
status_label = None
detail_label = None
log_text = None
loader_canvas = None

angle = 0


# =========================================================
#  HACKER LOG CONSOLE
# =========================================================
def log(msg: str):
    if log_text is None:
        return
    log_text.config(state="normal")
    prefix = random.choice(["[NODE]", "[MODEL]", "[SCAN]", "[BREAST]", "[CORE]"])
    log_text.insert("end", f"{prefix} {msg}\n")
    log_text.see("end")
    log_text.config(state="disabled")


# =========================================================
#  MATRIX SPINNING LOADER
# =========================================================
def start_loader():
    global animation_running, loader_canvas
    if animation_running:
        return

    animation_running = True

    loader_canvas = tk.Canvas(
        panel,
        width=200,
        height=200,
        bg=PANEL_BG,
        highlightthickness=0
    )
    loader_canvas.place(relx=0.5, rely=0.5, anchor="center")

    animate_loader()


def stop_loader():
    global animation_running, loader_canvas
    animation_running = False
    if loader_canvas:
        loader_canvas.destroy()
        loader_canvas = None


def animate_loader():
    global angle, animation_running
    if not animation_running or loader_canvas is None:
        return

    loader_canvas.delete("all")

    cx, cy = 100, 100
    r_outer = 60
    r_inner = 40

    loader_canvas.create_oval(
        cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
        outline=ACCENT, width=3
    )

    loader_canvas.create_arc(
        cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
        start=angle, extent=120,
        style="arc",
        width=6,
        outline=ACCENT
    )

    loader_canvas.create_oval(
        cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
        outline=MUTED, width=2
    )

    loader_canvas.create_text(
        cx, cy + 50,
        text="SCANNING...",
        fill=ACCENT,
        font=("Consolas", 10, "bold")
    )

    angle = (angle + 10) % 360
    loader_canvas.after(40, animate_loader)


# -----------------------------
# BUTTON HOVER EFFECT
# -----------------------------
def on_enter(e):
    e.widget["background"] = BTN_HOVER


def on_leave(e):
    e.widget["background"] = BTN_BG


# -----------------------------
# SHOW IMAGE
# -----------------------------
def show_image(img_path: Path):
    try:
        img = Image.open(img_path).resize((800, 500))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image:\n{e}")
        log(f"ERROR loading image -> {e}")


# -----------------------------
# IMAGE DETECTION (BREAST CANCER)
# -----------------------------
def detect_image():
    file_path = filedialog.askopenfilename(
        title="Select breast image (mammogram)",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    log(f"Image selected: {file_path}")
    status_label.config(text="STATUS: IMAGE SCAN ACTIVE")
    detail_label.config(text="DETAIL: YOLOv8 breast cancer model scanning...")
    start_loader()
    root.update_idletasks()

    try:
        # Run YOLO detection
        results = model.predict(file_path, save=True, conf=0.25, verbose=False)
        r0 = results[0]
        save_dir = Path(r0.save_dir)

        # Find saved annotated image
        stem = Path(file_path).stem
        candidates = list(save_dir.glob(f"{stem}.*"))

        if not candidates:
            candidates = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.png"))

        output_img = candidates[-1] if candidates else None
        if output_img:
            show_image(output_img)
        else:
            log("No annotated image found to display.")

        log("YOLO detection complete.")

        # Analyze detections
        tumor_detected = False
        best_conf_tumor = 0.0

        for box in r0.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0]) * 100  # to percentage

            # class 0 -> "1" (treated as NORMAL)
            # class 1 -> "tumor"
            if cls == 1:  # tumor
                tumor_detected = True
                if conf > best_conf_tumor:
                    best_conf_tumor = conf

        if not r0.boxes:  # No boxes at all
            status_label.config(text="STATUS: NO REGION DETECTED")
            detail_label.config(text="DETAIL: Model did not detect any bounding boxes.")
            log("No bounding boxes detected in image.")
        elif tumor_detected:
            status_label.config(text="STATUS: TUMOR DETECTED")
            detail_label.config(
                text=f"DETAIL: POSSIBLE TUMOR - CONFIDENCE ≈ {best_conf_tumor:.2f}%"
            )
            log(f"TUMOR detected with confidence ≈ {best_conf_tumor:.2f}%")

            # Popup alert
            messagebox.showwarning(
                "ALERT: POSSIBLE TUMOR",
                f"Tumor detected with approx. {best_conf_tumor:.2f}% confidence.\n\n"
                "This tool is NOT a medical diagnosis.\n"
                "Please consult a radiologist or doctor for professional evaluation."
            )
        else:
            status_label.config(text="STATUS: NO TUMOR DETECTED")
            detail_label.config(text="DETAIL: Only NORMAL regions detected.")
            log("No tumor regions detected. Classified as NORMAL.")

    except Exception as e:
        messagebox.showerror("Error", f"Detection failed:\n{e}")
        log(f"ERROR image detection -> {e}")

    finally:
        stop_loader()


# -----------------------------
# WEBCAM DETECTION (DEMO)
# -----------------------------
def run_webcam():
    """
    NOTE: Webcam breast cancer detection is NOT medically meaningful,
    this is only a live demo of the model running on frames.
    """
    global stop_webcam
    stop_webcam = False

    status_label.config(text="STATUS: LIVE DEMO FEED")
    detail_label.config(text="DETAIL: Webcam connected (demo only).")
    log("LIVE MODE: Webcam stream initializing for DEMO...")

    start_loader()
    root.update_idletasks()
    time.sleep(1)
    stop_loader()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not found.")
        log("ERROR: Webcam not detected.")
        return

    log("Webcam online. Streaming...")

    try:
        while not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                break

            # Run model on frame
            results = model(frame, verbose=False)[0]
            tumor_detected = False
            best_conf_tumor = 0.0

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                if cls == 1:  # tumor
                    tumor_detected = True
                    if conf > best_conf_tumor:
                        best_conf_tumor = conf

            # Draw predictions on frame
            frame_annotated = results.plot()
            frame_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((800, 500))
            img_tk = ImageTk.PhotoImage(img)
            panel.config(image=img_tk)
            panel.image = img_tk

            # Update status text
            if tumor_detected:
                status_label.config(
                    text=f"STATUS: TUMOR DETECTED (DEMO) ~{best_conf_tumor:.1f}%"
                )
                detail_label.config(text="DETAIL: Live demo – possible tumor pattern.")
            else:
                status_label.config(text="STATUS: NO TUMOR DETECTED (DEMO)")
                detail_label.config(text="DETAIL: Live demo – no tumor pattern seen.")

            root.update_idletasks()

    finally:
        cap.release()
        stop_webcam = True
        status_label.config(text="STATUS: FEED TERMINATED")
        detail_label.config(text="DETAIL: Demo stream closed.")
        log("Webcam demo stream terminated.")


def start_webcam():
    thread = threading.Thread(target=run_webcam)
    thread.daemon = True
    thread.start()


def stop_webcam_func():
    global stop_webcam
    stop_webcam = True
    log("Command received: Terminate feed")


# -----------------------------
# UI BUILDER (MATRIX UI)
# -----------------------------
root = tk.Tk()
root.title("MATRIX // BREAST CANCER DETECTION NODE // YOLOv8")

root.resizable(True, True)
root.state("zoomed")
root.configure(bg=BG)

# ---- Header ----
header = tk.Frame(root, bg=BG)
header.pack(fill="x", pady=(10, 0), padx=15)

tk.Label(
    header, text="MATRIX BREAST CANCER DETECTION CONSOLE",
    fg=ACCENT, bg=BG, font=("Consolas", 20, "bold")
).pack(side="left")

tk.Label(
    header,
    text="MODE: ONLINE  //  ENGINE: YOLOv8  //  NODE STATUS: ACTIVE",
    fg=MUTED, bg=BG, font=("Consolas", 9)
).pack(side="left", padx=(15, 0))

accent_line = tk.Frame(root, bg=ACCENT, height=2)
accent_line.pack(fill="x", padx=15, pady=(8, 10))


# ---- Main Layout ----
main = tk.Frame(root, bg=BG)
main.pack(fill="both", expand=True, padx=15, pady=10)

# Sidebar
sidebar = tk.Frame(main, bg=SIDEBAR_BG, width=280)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

tk.Label(
    sidebar, text="> CONTROL PANEL",
    fg=ACCENT, bg=SIDEBAR_BG, font=("Consolas", 14, "bold")
).pack(pady=(20, 5), anchor="w", padx=20)

btn_box = tk.Frame(sidebar, bg=SIDEBAR_BG)
btn_box.pack(padx=20, pady=10, fill="x")


def make_btn(text, command):
    btn = tk.Button(
        btn_box, text=text, command=command,
        fg=TEXT, bg=BTN_BG, activebackground=BTN_HOVER,
        bd=0, relief="flat", font=("Consolas", 11), pady=6
    )
    btn.pack(fill="x", pady=6)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


make_btn("▣  IMAGE SCAN", detect_image)
make_btn("▣  LIVE DEMO (WEBCAM)", start_webcam)
make_btn("✘  TERMINATE FEED", stop_webcam_func)

# ---- System Log ----
tk.Label(
    sidebar, text="\n> SYSTEM LOG",
    fg=ACCENT, bg=SIDEBAR_BG, font=("Consolas", 12, "bold")
).pack(anchor="w", padx=20)

log_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
log_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))

log_text = tk.Text(
    log_frame, bg="#000000", fg=ACCENT, insertbackground=ACCENT,
    font=("Consolas", 9), relief="flat", wrap="none"
)
log_text.pack(fill="both", expand=True)
log_text.config(state="disabled")

log("BOOT: Breast cancer detection console online.")
log("YOLOv8 model loaded. Awaiting mammogram image for analysis.")


# ---- Preview Section ----
content = tk.Frame(main, bg=BG)
content.pack(side="left", fill="both", expand=True, padx=(20, 0))

preview_outer = tk.Frame(content, bg=ACCENT_SOFT)
preview_outer.pack(pady=(5, 10))

preview_inner = tk.Frame(preview_outer, bg=PANEL_BG)
preview_inner.pack(padx=2, pady=2)

panel = tk.Label(preview_inner, bg=PANEL_BG, width=800, height=500)
panel.pack()

# ---- Status Bar ----
status_frame = tk.Frame(content, bg=BG)
status_frame.pack(fill="x")

status_label = tk.Label(
    status_frame, text="STATUS: IDLE",
    fg=ACCENT, bg=BG, font=("Consolas", 11, "bold")
)
status_label.pack(fill="x")

detail_label = tk.Label(
    status_frame, text="DETAIL: Awaiting breast image input...",
    fg=MUTED, bg=BG, font=("Consolas", 9)
)
detail_label.pack(fill="x")

root.mainloop()
