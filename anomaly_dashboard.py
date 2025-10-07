#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anomaly_dashboard.py

GUI dashboard for a two-stage cascade:
  1) Autoencoder reconstruction gate
  2) YOLOv8 person cue (fast semantic hint)

Includes:
- Stable model loading (weights_only fallback)
- CUDA autocast + inference_mode
- Optional batched AE for big speedups
- Clean pagination to avoid X BadAlloc
- View-selected image in a separate process with optional YOLO overlay
- CSV export (with file dialog)

Author: Tayyab Rehman
"""

import os
import sys
import csv
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import multiprocessing
from multiprocessing import set_start_method
from pathlib import Path
from typing import Iterable, List, Tuple

# ----------------------------
# 1. CONFIG
# ----------------------------

AUTOENCODER_MODEL_PATH = 'autoencoder_model.pth'
YOLO_MODEL_PATH = 'yolov8n.pt'

RECONSTRUCTION_THRESHOLD = 0.0015     # tune per camera/site
CONFIDENCE_THRESHOLD = 0.45           # YOLO person threshold

IMAGE_DIR = Path("/home/user/Desktop/AAMS 25/archive")  # <- dataset root
IMAGE_RESIZE = (128, 128)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

PAGE_SIZE = 200                       # GUI rows per page
AE_BATCH = 32                         # GPU batch; set to 8–16 on CPU
USE_BATCHED_AE = True                 # turn off to run single-frame AE

# If you trained AE with normalization, update this transform accordingly.
TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_RESIZE),
    transforms.ToTensor(),           # 0..1, matches decoder Sigmoid
])

# ----------------------------
# 2. MODEL
# ----------------------------

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # keep if training used 0..1 reconstruction loss
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ----------------------------
# 3. HELPERS
# ----------------------------

def iter_image_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Base dir not found: {root}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def safe_open_rgb(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[warn] skipping unreadable image: {path} -> {e}")
        return None

def batched_indices(n: int, bs: int) -> Iterable[List[int]]:
    i = 0
    while i < n:
        j = min(i + bs, n)
        yield list(range(i, j))
        i = j

def show_image_in_new_process(image_path, window_title, yolo_model_path, conf_thresh):
    """
    Separate-process viewer: loads YOLO fresh in the child, draws boxes if 'person' over conf.
    Keeps main GUI light and avoids GPU/handle contention.
    """
    try:
        model = YOLO(yolo_model_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Process Error] Could not open image at {image_path}")
            return
        results = model(img, conf=conf_thresh, verbose=False)

        draw = False
        if len(results) > 0 and results[0].boxes is not None:
            names = model.names
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if names.get(cls_id, "") == "person":
                    draw = True
                    break

        annotated = results[0].plot() if draw else img
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[Process Error] Could not display image: {e}")

# ----------------------------
# 4. GUI APP
# ----------------------------

class AnomalyDashboard(tk.Tk):
    def __init__(self, device: torch.device):
        super().__init__()
        self.title("Combined Anomaly Detection Dashboard")
        self.geometry("1100x740")
        self.device = device

        self.autoencoder_model: nn.Module = None
        self.yolo_model = None

        # Data holders
        self.events_all: List[dict] = []  # events after grouping
        self.current_page = 0

        # --- Top controls ---
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        self.run_button = ttk.Button(top, text="Run Full Analysis", command=self.start_analysis_thread)
        self.run_button.pack(side=tk.LEFT)

        self.status_label = ttk.Label(top, text="Ready.", font=("Helvetica", 10, "italic"))
        self.status_label.pack(side=tk.LEFT, padx=10)

        export_btn = ttk.Button(top, text="Export CSV", command=self.export_csv)
        export_btn.pack(side=tk.RIGHT)

        # --- Table ---
        table_frame = ttk.Frame(self, padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(table_frame, columns=("type","frames","error","path"), show="headings")
        self.tree.heading("type", text="Event Type")
        self.tree.heading("frames", text="Frames")
        self.tree.heading("error", text="Recon Error")
        self.tree.heading("path", text="File")
        self.tree.column("type", width=160, anchor="w")
        self.tree.column("frames", width=80, anchor="center")
        self.tree.column("error", width=120, anchor="center")
        self.tree.column("path", width=720, anchor="w")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Bottom ---
        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill=tk.X)

        self.prev_btn = ttk.Button(bottom, text="⟵ Prev", command=self.prev_page, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT)

        self.page_label = ttk.Label(bottom, text="Page 0 / 0")
        self.page_label.pack(side=tk.LEFT, padx=10)

        self.next_btn = ttk.Button(bottom, text="Next ⟶", command=self.next_page, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT)

        self.view_btn = ttk.Button(bottom, text="View Selected", command=self.view_selected, state=tk.DISABLED)
        self.view_btn.pack(side=tk.RIGHT)

        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<Double-1>", lambda _e: self.view_selected())

    # ---------- Model Loading ----------
    def load_models(self) -> bool:
        try:
            self.status_label.config(text="Loading Autoencoder...")
            self.autoencoder_model = ConvAutoencoder().to(self.device)

            # robust load across PyTorch versions
            try:
                sd = torch.load(AUTOENCODER_MODEL_PATH, map_location=self.device, weights_only=True)
            except TypeError:
                sd = torch.load(AUTOENCODER_MODEL_PATH, map_location=self.device)
            self.autoencoder_model.load_state_dict(sd)
            self.autoencoder_model.eval()

            # warmup
            _ = self.autoencoder_model(torch.zeros(1,3,IMAGE_RESIZE[1],IMAGE_RESIZE[0], device=self.device))
            print("Autoencoder loaded.")

            self.status_label.config(text="Loading YOLO...")
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            _ = self.yolo_model(np.zeros((64,64,3), dtype=np.uint8), verbose=False)
            print("YOLO model loaded.")

            self.status_label.config(text="Models ready.")
            return True
        except Exception as e:
            self.status_label.config(text=f"Error loading models: {e}")
            print(f"Error during model loading: {e}")
            return False

    # ---------- Analysis Thread ----------
    def start_analysis_thread(self):
        self.run_button.config(state=tk.DISABLED)
        self.view_btn.config(state=tk.DISABLED)
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
        self.tree.delete(*self.tree.get_children())
        t = threading.Thread(target=self.run_full_analysis, daemon=True)
        t.start()

    def run_full_analysis(self):
        if self.autoencoder_model is None or self.yolo_model is None:
            if not self.load_models():
                self.after(0, self.analysis_failed, "Could not load models.")
                return

        self.after(0, lambda: self.status_label.config(text="Scanning images..."))

        paths = sorted(iter_image_paths(IMAGE_DIR), key=lambda p: str(p))
        if not paths:
            self.after(0, self.analysis_failed,
                       f"No images found under '{IMAGE_DIR}'. Supported: {sorted(IMG_EXTS)}")
            return

        all_events: List[dict] = []
        device_is_cuda = (self.device.type == 'cuda')

        if USE_BATCHED_AE:
            tensors = []
            items: List[Tuple[Path, Image.Image]] = []
            for p in paths:
                pil_im = safe_open_rgb(p)
                if pil_im is None:
                    continue
                x = TRANSFORM(pil_im)
                tensors.append(x)
                items.append((p, pil_im))

            mse_reduce_mean = lambda yb, xb: ((yb - xb) ** 2).mean(dim=(1,2,3))

            for chunk in tqdm(batched_indices(len(items), AE_BATCH), desc="AE Batches"):
                xb = torch.stack([tensors[i] for i in chunk], dim=0).to(self.device, non_blocking=True)
                with torch.inference_mode():
                    if device_is_cuda:
                        with torch.cuda.amp.autocast():
                            yb = self.autoencoder_model(xb)
                    else:
                        yb = self.autoencoder_model(xb)
                errs = mse_reduce_mean(yb, xb).detach().cpu().numpy().tolist()

                for j, idx in enumerate(chunk):
                    err = float(errs[j])
                    if err <= RECONSTRUCTION_THRESHOLD:
                        continue
                    p, pil_im = items[idx]
                    frame_bgr = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                    res = self.yolo_model(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)

                    is_person = False
                    if len(res) > 0 and res[0].boxes is not None:
                        names = self.yolo_model.names
                        for box in res[0].boxes:
                            cls_id = int(box.cls[0])
                            if names.get(cls_id, "") == "person":
                                is_person = True
                                break

                    event_type = "Person (YOLO)" if is_person else "Anomaly (Autoencoder)"
                    all_events.append({"path": str(p), "type": event_type, "error": err})
        else:
            mse = nn.MSELoss()
            for p in tqdm(paths, desc="Analyzing Frames"):
                pil_im = safe_open_rgb(p)
                if pil_im is None:
                    continue

                x = TRANSFORM(pil_im).unsqueeze(0).to(self.device, non_blocking=True)
                with torch.inference_mode():
                    if device_is_cuda:
                        with torch.cuda.amp.autocast():
                            y = self.autoencoder_model(x)
                    else:
                        y = self.autoencoder_model(x)
                err = mse(y, x).item()

                if err > RECONSTRUCTION_THRESHOLD:
                    frame_bgr = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                    res = self.yolo_model(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)

                    is_person = False
                    if len(res) > 0 and res[0].boxes is not None:
                        names = self.yolo_model.names
                        for box in res[0].boxes:
                            cls_id = int(box.cls[0])
                            if names.get(cls_id, "") == "person":
                                is_person = True
                                break

                    event_type = "Person (YOLO)" if is_person else "Anomaly (Autoencoder)"
                    all_events.append({"path": str(p), "type": event_type, "error": err})

        # Group consecutive same-type events into bursts
        grouped: List[List[dict]] = []
        if all_events:
            cur = [all_events[0]]
            for i in range(1, len(all_events)):
                if all_events[i]["type"] == all_events[i-1]["type"]:
                    cur.append(all_events[i])
                else:
                    grouped.append(cur)
                    cur = [all_events[i]]
            grouped.append(cur)

        final_events = []
        for g in grouped:
            rep = dict(g[0])
            rep["duration_frames"] = len(g)
            final_events.append(rep)

        self.events_all = final_events
        self.current_page = 0
        self.after(0, self.refresh_page)
        self.after(0, lambda: self.status_label.config(
            text=f"Analysis complete. {len(paths)} images scanned, {len(final_events)} events.")
        )
        self.after(0, lambda: self.run_button.config(state=tk.NORMAL))

    def analysis_failed(self, msg):
        self.status_label.config(text=f"Error: {msg}")
        self.run_button.config(state=tk.NORMAL)

    # ---------- Pagination / Table ----------
    def refresh_page(self):
        self.tree.delete(*self.tree.get_children())
        n = len(self.events_all)
        if n == 0:
            self.page_label.config(text="Page 0 / 0")
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.view_btn.config(state=tk.DISABLED)
            return

        pages = (n + PAGE_SIZE - 1) // PAGE_SIZE
        self.current_page = max(0, min(self.current_page, pages - 1))
        start = self.current_page * PAGE_SIZE
        end = min(start + PAGE_SIZE, n)
        page_rows = self.events_all[start:end]

        for ev in page_rows:
            self.tree.insert("", "end", values=(
                ev["type"],
                ev.get("duration_frames", 1),
                f"{ev['error']:.6f}",
                os.path.basename(ev["path"]),
            ), tags=(ev["path"],))

        self.page_label.config(text=f"Page {self.current_page+1} / {pages}")
        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_page < pages - 1 else tk.DISABLED)
        self.view_btn.config(state=tk.DISABLED)

    def next_page(self):
        self.current_page += 1
        self.refresh_page()

    def prev_page(self):
        self.current_page -= 1
        self.refresh_page()

    def on_select(self, _evt):
        sel = self.tree.selection()
        self.view_btn.config(state=tk.NORMAL if sel else tk.DISABLED)

    # ---------- Actions ----------
    def view_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        tags = item.get("tags", [])
        if not tags:
            messagebox.showerror("Error", "Could not resolve file path.")
            return
        full_path = None
        for t in tags:
            if t.startswith("/"):  # absolute paths on Unix
                full_path = t
                break
        if not full_path:
            messagebox.showerror("Error", "Could not resolve file path.")
            return

        window_title = f"{item['values'][0]} @ {item['values'][3]}"
        p = multiprocessing.Process(
            target=show_image_in_new_process,
            kwargs=dict(
                image_path=full_path,
                window_title=window_title,
                yolo_model_path=YOLO_MODEL_PATH,
                conf_thresh=CONFIDENCE_THRESHOLD,
            ),
        )
        p.daemon = True
        p.start()

    def export_csv(self):
        if not self.events_all:
            messagebox.showinfo("Export CSV", "No events to export.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save events CSV",
            defaultextension=".csv",
            initialdir=str(IMAGE_DIR),
            filetypes=[("CSV","*.csv")]
        )
        if not out_path:
            return
        try:
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path","type","reconstruction_error","duration_frames"])
                for ev in self.events_all:
                    w.writerow([ev["path"], ev["type"], f"{ev['error']:.8f}", ev.get("duration_frames", 1)])
            messagebox.showinfo("Export CSV", f"Saved: {out_path}")
        except Exception as e:
            messagebox.showerror("Export CSV", f"Failed to save CSV: {e}")

# ----------------------------
# 5. MAIN
# ----------------------------

if __name__ == "__main__":
    # Windows/mac safety for multiprocessing and crisp UI on Windows
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass

    try:
        set_start_method('spawn', force=True)
        print("Process start method set to 'spawn'.")
    except RuntimeError:
        print("Process start method was already set.")

    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Windows DPI fix
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    app = AnomalyDashboard(device)
    app.mainloop()
