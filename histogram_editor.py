# histogram_editor.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import numpy as np
import threading
import glob
import os
import random
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from backend import get_base_histogram

class HistogramEditor(tk.Toplevel):
    def __init__(self, parent, input_dir, advanced_params, on_save_callback):
        super().__init__(parent)
        self.transient(parent)
        self.title("Custom Histogram Editor")
        self.geometry("850x650") 
        self.parent = parent
        self.input_dir = input_dir
        self.advanced_params = advanced_params
        self.on_save_callback = on_save_callback

        # --- State Variables ---
        self.hist_fixed = None
        self.hist_shufflable = None
        self.current_permutation = list(range(self.advanced_params['hist_chunks']))
        self.saved_permutations = [] 
        
        self.drag_source_chunk = None
        self.drop_indicator_line = None
        self.shuffle_dark_var = tk.BooleanVar(value=False)

        # --- Main Layout ---
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)

        # --- Left Panel: Plot ---
        plot_frame = ttk.LabelFrame(main_frame, text="Histogram Preview (Drag chunks to reorder)", padding=10)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.canvas.mpl_connect('button_press_event', self.on_plot_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_plot_motion)
        self.canvas.mpl_connect('button_release_event', self.on_plot_release)

        # --- Right Panel: Controls ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew")

        # 1. Source Selection
        source_frame = ttk.LabelFrame(controls_frame, text="1. Load Base Histogram", padding=10)
        source_frame.pack(fill="x", pady=(0, 10))
        
        self.source_var = tk.StringVar(value="random")
        ttk.Radiobutton(source_frame, text="From Random Image", variable=self.source_var, value="random").pack(anchor="w")
        ttk.Radiobutton(source_frame, text="From Dataset Mean", variable=self.source_var, value="mean").pack(anchor="w")
        ttk.Radiobutton(source_frame, text="From Selected Images", variable=self.source_var, value="select").pack(anchor="w")
        
        ttk.Checkbutton(source_frame, text="Include darkest chunk in shuffling", variable=self.shuffle_dark_var, command=self.load_histogram_thread).pack(anchor="w", pady=(5,0))
        
        ttk.Button(source_frame, text="Load Histogram", command=self.load_histogram_thread).pack(pady=5, fill="x")

        # 2. Permutation Controls
        perm_frame = ttk.LabelFrame(controls_frame, text="2. Current Order", padding=10)
        perm_frame.pack(fill="both", expand=True, pady=10)
        
        self.perm_listbox = tk.Listbox(perm_frame, height=8)
        self.perm_listbox.pack(fill="both", expand=True, pady=5)
        
        reset_button = ttk.Button(perm_frame, text="Reset to Default", command=self.reset_permutation)
        reset_button.pack(fill="x")

        # 3. Save Controls
        save_frame = ttk.LabelFrame(controls_frame, text="3. Manage Contrasts", padding=10)
        save_frame.pack(fill="x", pady=10)
        
        add_button = ttk.Button(save_frame, text="Add Current Permutation to List", command=self.add_permutation)
        add_button.pack(fill="x", pady=5)
        self.saved_count_label = ttk.Label(save_frame, text="Saved contrasts: 0")
        self.saved_count_label.pack()

        # --- Bottom Panel: Actions ---
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(action_frame, text="Save & Close", command=self.save_and_close).pack(side="right", padx=5)
        ttk.Button(action_frame, text="Cancel", command=self.destroy).pack(side="right")
        
        self.status_label = ttk.Label(self, text="Select a source and load a histogram to begin.", relief="sunken")
        self.status_label.pack(side="bottom", fill="x")

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def load_histogram_thread(self):
        thread = threading.Thread(target=self.load_histogram)
        thread.daemon = True
        thread.start()

    def load_histogram(self):
        source = self.source_var.get()
        image_paths = []

        if not self.input_dir or not os.path.isdir(self.input_dir):
            self.update_status("Error: Main input directory not set.")
            messagebox.showerror("Error", "Please set the main input directory in the main window first.", parent=self)
            return

        all_files = []
        supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        for ext in supported_extensions:
            all_files.extend(glob.glob(os.path.join(self.input_dir, ext)))

        if not all_files:
            self.update_status("Error: No images found in the input directory.")
            messagebox.showerror("Error", f"No images found in {self.input_dir}", parent=self)
            return

        if source == "random":
            image_paths = [random.choice(all_files)]
        elif source == "mean":
            image_paths = all_files
        elif source == "select":
            selected_paths = filedialog.askopenfilenames(
                title="Select images for histogram",
                initialdir=self.input_dir,
                filetypes=[("Image Files", ".png .jpg .jpeg")]
            )
            if not selected_paths:
                self.update_status("Image selection cancelled.")
                return
            image_paths = list(selected_paths)
        
        self.update_status(f"Loading from {len(image_paths)} image(s)...")
        
        # *** Pass the checkbox state to the backend ***
        hist_fixed, hist_shufflable, msg = get_base_histogram(
            image_paths,
            self.advanced_params['num_bins'],
            self.advanced_params['dark_threshold'],
            self.update_status,
            shuffle_dark_chunk=self.shuffle_dark_var.get()
        )

        if hist_fixed is not None:
            self.hist_fixed = hist_fixed
            self.hist_shufflable = hist_shufflable
            self.reset_permutation()
        else:
            self.update_status(f"Error: {msg}")
            messagebox.showerror("Error", f"Failed to calculate histogram:\n{msg}", parent=self)

    def draw_histogram(self):
        if self.hist_shufflable is None: return
        
        self.ax.clear()
        
        num_chunks = self.advanced_params['hist_chunks']
        num_bins = self.advanced_params['num_bins']
        chunk_size = num_bins // num_chunks
        
        shufflable_np = self.hist_shufflable.numpy().flatten()
        shufflable_chunks = shufflable_np.reshape((num_chunks, chunk_size))
        permuted_chunks = shufflable_chunks[self.current_permutation, :]

        fixed_np = self.hist_fixed.numpy().flatten()
        
        colors = plt.cm.get_cmap('viridis', num_chunks)

        for i in range(num_chunks):
            original_chunk_idx = self.current_permutation[i]
            start_bin = i * chunk_size
            end_bin = start_bin + chunk_size
            
            chunk_data = permuted_chunks[i, :] + fixed_np[start_bin:end_bin]
            x_positions = range(start_bin, end_bin)
            
            alpha = 0.5 if i == self.drag_source_chunk else 1.0
            
            self.ax.bar(x_positions, chunk_data, color=colors(original_chunk_idx), width=1.0, alpha=alpha, zorder=5)

        self.ax.set_title("Histogram Preview")
        self.ax.set_xlabel("Intensity Bin")
        self.ax.set_ylabel("Pixel Count (Averaged)")
        self.ax.set_yticklabels([])
        self.ax.set_xlim(0, num_bins)
        self.canvas.draw()
        self.update_listbox(colors)

    def update_listbox(self, colors):
        self.perm_listbox.delete(0, tk.END)
        for i, chunk_idx in enumerate(self.current_permutation):
            self.perm_listbox.insert(tk.END, f"Position {i+1}: (Original Chunk {chunk_idx+1})")
            rgba_color = colors(chunk_idx, bytes=True)
            hex_color = f'#{rgba_color[0]:02x}{rgba_color[1]:02x}{rgba_color[2]:02x}'
            self.perm_listbox.itemconfig(i, {'bg': hex_color})

    def reset_permutation(self):
        self.current_permutation = list(range(self.advanced_params['hist_chunks']))
        self.draw_histogram()

    def _get_chunk_from_x(self, x_data):
        if x_data is None or self.hist_shufflable is None: return None
        num_chunks = self.advanced_params['hist_chunks']
        num_bins = self.advanced_params['num_bins']
        chunk_size = num_bins // num_chunks
        x_data = max(0, min(x_data, num_bins - 1))
        return int(x_data // chunk_size)

    def on_plot_press(self, event):
        if event.inaxes != self.ax: return
        chunk_index = self._get_chunk_from_x(event.xdata)
        if chunk_index is not None:
            self.drag_source_chunk = chunk_index
            self.canvas.get_tk_widget().config(cursor="hand2")
            self.draw_histogram()

    def on_plot_motion(self, event):
        if self.drag_source_chunk is None or event.inaxes != self.ax: return
        target_chunk_index = self._get_chunk_from_x(event.xdata)
        if target_chunk_index is None: return
        chunk_size = self.advanced_params['num_bins'] // self.advanced_params['hist_chunks']
        drop_position = target_chunk_index * chunk_size
        if self.drop_indicator_line:
            self.drop_indicator_line.set_xdata([drop_position, drop_position])
        else:
            self.drop_indicator_line = self.ax.axvline(drop_position, color='red', linestyle='--', linewidth=2, zorder=10)
        self.canvas.draw_idle()

    def on_plot_release(self, event):
        if self.drag_source_chunk is None: return
        self.canvas.get_tk_widget().config(cursor="")
        if self.drop_indicator_line:
            self.drop_indicator_line.remove()
            self.drop_indicator_line = None
        drop_chunk_index = self._get_chunk_from_x(event.xdata)
        if drop_chunk_index is not None and self.drag_source_chunk != drop_chunk_index:
            item_to_move = self.current_permutation.pop(self.drag_source_chunk)
            self.current_permutation.insert(drop_chunk_index, item_to_move)
        self.drag_source_chunk = None
        self.draw_histogram()

    def add_permutation(self):
        if self.hist_shufflable is None:
            messagebox.showerror("Error", "Please load a base histogram first.", parent=self)
            return
        
        #  Save the permutation along with the checkbox state 
        perm_data = (torch.tensor([self.current_permutation], dtype=torch.long), self.shuffle_dark_var.get())
        self.saved_permutations.append(perm_data)
        
        self.saved_count_label.config(text=f"Saved contrasts: {len(self.saved_permutations)}")
        self.update_status(f"Permutation added. Total: {len(self.saved_permutations)}")

    def save_and_close(self):
        if self.on_save_callback:
            self.on_save_callback(self.saved_permutations)
        self.destroy()
