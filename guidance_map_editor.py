# guidance_map_editor.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import numpy as np
import os
import glob
from PIL import Image

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GuidanceMapEditor(tk.Toplevel):
    def __init__(self, parent, input_dir, advanced_params, on_save_callback):
        super().__init__(parent)
        self.transient(parent)
        self.title("Visual Guidance Editor")
        self.geometry("900x700")
        self.parent = parent
        self.input_dir = input_dir
        self.advanced_params = advanced_params
        self.on_save_callback = on_save_callback

        # --- State Variables ---
        self.original_image_np = None # Normalized [0, 1]
        self.preview_image_np = None  # Normalized [0, 1]
        self.selection_mask = None
        self.selection_overlay = None
        self.committed_changes = [] # List of (source_range, target_value)

        # --- UI Variables ---
        self.tolerance = tk.DoubleVar(value=0.05)
        self.target_intensity = tk.DoubleVar(value=0.5)

        # --- Main Layout ---
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3) # Image
        main_frame.columnconfigure(1, weight=1) # Controls

        # --- Left Panel: Image Canvas ---
        plot_frame = ttk.LabelFrame(main_frame, text="Image Preview (Click to select intensity)", padding=5)
        plot_frame.grid(row=0, column=0, sticky="nsew", rowspan=2)
        fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.axis('off')
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_image_click)

        # --- Right Panel: Controls ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=10)

        # 1. Image Selection
        ttk.Button(controls_frame, text="Select Source Image...", command=self.load_image).pack(fill="x", pady=5)
        
        # 2. Selection Controls
        sel_frame = ttk.LabelFrame(controls_frame, text="Selection Controls", padding=10)
        sel_frame.pack(fill="x", pady=10)
        
        ttk.Label(sel_frame, text="Tolerance:").grid(row=0, column=0, sticky="w")
        self.tol_label = ttk.Label(sel_frame, text=f"{self.tolerance.get():.2f}")
        self.tol_label.grid(row=0, column=2, sticky="e")
        s_tol = ttk.Scale(sel_frame, from_=0.0, to=1.0, variable=self.tolerance, command=self.update_tolerance_label)
        s_tol.grid(row=0, column=1, sticky="ew")

        ttk.Label(sel_frame, text="Target Intensity:").grid(row=1, column=0, sticky="w")
        self.tgt_label = ttk.Label(sel_frame, text=f"{self.target_intensity.get():.2f}")
        self.tgt_label.grid(row=1, column=2, sticky="e")
        s_tgt = ttk.Scale(sel_frame, from_=0.0, to=1.0, variable=self.target_intensity, command=self.update_target_label)
        s_tgt.grid(row=1, column=1, sticky="ew")
        
        # 3. Actions
        act_frame = ttk.LabelFrame(controls_frame, text="Actions", padding=10)
        act_frame.pack(fill="x", pady=10)
        ttk.Button(act_frame, text="Commit Change", command=self.commit_change).pack(fill="x", pady=5)
        ttk.Button(act_frame, text="Reset All Changes", command=self.reset_all).pack(fill="x", pady=5)
        self.changes_label = ttk.Label(act_frame, text="Committed changes: 0")
        self.changes_label.pack(pady=5)
        
        # --- Bottom Panel: Save/Cancel ---
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(action_frame, text="Create Contrast & Close", command=self.save_and_close, style="Accent.TButton").pack(side="right", padx=5)
        ttk.Button(action_frame, text="Cancel", command=self.destroy).pack(side="right")
        self.status_label = ttk.Label(self, text="Please select a source image to begin.", relief="sunken")
        self.status_label.pack(side="bottom", fill="x")

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select a template image",
            initialdir=self.input_dir,
            filetypes=[("Image Files", ".png .jpg .jpeg")]
        )
        if not path: return
        
        try:
            with Image.open(path).convert('L') as img:
                img_np = np.array(img, dtype=np.float32)
            self.original_image_np = img_np / 255.0
            self.preview_image_np = self.original_image_np.copy()
            self.selection_mask = None
            self.committed_changes = []
            self.changes_label.config(text="Committed changes: 0")
            self.draw_image()
            self.update_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}", parent=self)
            self.update_status("Error loading image.")

    def draw_image(self):
        if self.preview_image_np is None: return
        self.ax.clear()
        self.ax.imshow(self.preview_image_np, cmap='gray', vmin=0, vmax=1)
        
        if self.selection_mask is not None:
            # Create a red overlay for the selection
            overlay = np.zeros((*self.selection_mask.shape, 4)) # RGBA
            overlay[self.selection_mask] = [1, 0, 0, 0.4] # Red, 40% opacity
            self.ax.imshow(overlay)

        self.ax.axis('off')
        self.canvas.draw()
        
    def update_tolerance_label(self, val):
        self.tol_label.config(text=f"{float(val):.2f}")

    def update_target_label(self, val):
        self.tgt_label.config(text=f"{float(val):.2f}")

    def on_image_click(self, event):
        if event.inaxes != self.ax or self.original_image_np is None: return
        
        y, x = int(event.ydata), int(event.xdata)
        clicked_intensity = self.original_image_np[y, x]
        
        tolerance_val = self.tolerance.get()
        lower_bound = clicked_intensity - tolerance_val
        upper_bound = clicked_intensity + tolerance_val
        
        self.selection_mask = (self.original_image_np >= lower_bound) & (self.original_image_np <= upper_bound)
        
        # Store the range for commit
        self.last_selection_range = (lower_bound, upper_bound)
        self.update_status(f"Selected pixels with intensity ~{clicked_intensity:.2f} (±{tolerance_val:.2f})")
        self.draw_image()

    def commit_change(self):
        if self.selection_mask is None:
            messagebox.showwarning("Warning", "No pixels selected. Click on the image first.", parent=self)
            return
        
        target_val = self.target_intensity.get()
        self.preview_image_np[self.selection_mask] = target_val
        self.committed_changes.append((self.last_selection_range, target_val))
        
        self.selection_mask = None
        self.draw_image()
        self.update_status(f"Change committed. Mapped range {self.last_selection_range[0]:.2f}-{self.last_selection_range[1]:.2f} to {target_val:.2f}")
        self.changes_label.config(text=f"Committed changes: {len(self.committed_changes)}")

    def reset_all(self):
        if self.original_image_np is None: return
        self.preview_image_np = self.original_image_np.copy()
        self.selection_mask = None
        self.committed_changes = []
        self.draw_image()
        self.changes_label.config(text="Committed changes: 0")
        self.update_status("All changes have been reset.")

    def save_and_close(self):
        if not self.committed_changes:
            messagebox.showerror("Error", "No changes have been committed.", parent=self)
            return

        try:
            num_chunks = self.advanced_params['hist_chunks']
            
            # 1. Initialize mapping arrays
            new_perm = [-1] * num_chunks
            
            # 2. Determine chunk mappings from user changes
            source_chunks_mapped = set()
            target_chunks_mapped = set()
            
            for source_range, target_val in self.committed_changes:
                source_mid_point = (source_range[0] + source_range[1]) / 2.0
                source_chunk = min(num_chunks - 1, int(source_mid_point * num_chunks))
                target_chunk = min(num_chunks - 1, int(target_val * num_chunks))
                
                # If a source chunk is re-mapped, the latest one wins
                if source_chunk in source_chunks_mapped:
                    old_target = new_perm[source_chunk]
                    if old_target != -1: target_chunks_mapped.discard(old_target)

                new_perm[source_chunk] = target_chunk
                source_chunks_mapped.add(source_chunk)
                target_chunks_mapped.add(target_chunk)

            # 3. Fill in unmapped chunks to create a valid permutation
            unmapped_sources = sorted([i for i in range(num_chunks) if i not in source_chunks_mapped])
            unmapped_targets = sorted([i for i in range(num_chunks) if i not in target_chunks_mapped])

            if len(unmapped_sources) != len(unmapped_targets):
                for src_chunk in unmapped_sources:
                    if unmapped_targets:
                        new_perm[src_chunk] = unmapped_targets.pop(0)
                    else:
                        new_perm[src_chunk] = src_chunk
            else:
                 for src, tgt in zip(unmapped_sources, unmapped_targets):
                    new_perm[src] = tgt

            if -1 in new_perm or len(set(new_perm)) != num_chunks:
                raise ValueError("Failed to create a valid one-to-one permutation mapping.")

            final_permutation = (torch.tensor([new_perm], dtype=torch.long), True) 
            
            if self.on_save_callback:
                self.on_save_callback([final_permutation])
            self.destroy()

        except Exception as e:
            messagebox.showerror("Error Creating Permutation", f"An error occurred: {e}", parent=self)
            self.update_status(f"Error: {e}")