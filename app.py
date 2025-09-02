# app.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os

from backend import generate_contrasts
from histogram_editor import HistogramEditor
from guidance_map_editor import GuidanceMapEditor

# =====================================================================================
# ## Helper Class for Tooltips
# =====================================================================================
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget; self.text = text; self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip); self.widget.bind("<Leave>", self.hide_tooltip)
    def show_tooltip(self, event):
        x, y, _, _ = self.widget.bbox("insert"); x += self.widget.winfo_rootx() + 25; y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(self.widget); self.tooltip_window.wm_overrideredirect(True); self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip_window, text=self.text, justify='left', background="#00517D", relief='solid', borderwidth=1, font=("Helvetica", "10", "normal"), foreground="white")
        label.pack(ipadx=1)
    def hide_tooltip(self, event):
        if self.tooltip_window: self.tooltip_window.destroy(); self.tooltip_window = None

# =====================================================================================
# ## Main Application
# =====================================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MRI Contrast Synthesizer")
        self.geometry("680x800")

        # --- UI Variables ---
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.generation_mode = tk.StringVar(value="random")

        self.num_contrasts = tk.StringVar(value="5")
        self.fixed_contrasts = tk.BooleanVar(value=True)

        self.num_bins = tk.StringVar(value="240")
        self.hist_chunks = tk.StringVar(value="8")
        self.dark_threshold = tk.DoubleVar(value=0.15)

        self.defined_contrasts = []

        # --- UI Layout ---
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)

        # --- FOLDERS ---
        ttk.Label(main_frame, text="Input Dataset Directory:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,5))
        ttk.Entry(main_frame, textvariable=self.input_dir).grid(row=1, column=0, sticky="ew", columnspan=2)
        ttk.Button(main_frame, text="Browse...", command=self.select_input_dir).grid(row=1, column=2, padx=5)
        
        ttk.Label(main_frame, text="Output Directory:", font=("Helvetica", 10, "bold")).grid(row=2, column=0, columnspan=2, sticky="w", pady=(15, 5))
        self.output_entry = ttk.Entry(main_frame, textvariable=self.output_dir)
        self.output_entry.grid(row=3, column=0, sticky="ew", columnspan=2)
        self.output_button = ttk.Button(main_frame, text="Browse...", command=self.select_output_dir)
        self.output_button.grid(row=3, column=2, padx=5)

        # --- GENERATION MODE ---
        mode_frame = ttk.LabelFrame(main_frame, text="Generation Mode", padding=10)
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=20)
        ttk.Radiobutton(mode_frame, text="Random Generation", variable=self.generation_mode, value="random", command=self.toggle_ui_mode).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Custom Histogram Generation", variable=self.generation_mode, value="custom", command=self.toggle_ui_mode).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Visual Guidance Editor", variable=self.generation_mode, value="visual", command=self.toggle_ui_mode).pack(anchor="w") 

        # --- MODE-SPECIFIC FRAMES ---

        self.random_mode_frame = ttk.LabelFrame(main_frame, text="Random Mode Options", padding="10")
        self.random_mode_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        self.random_mode_frame.columnconfigure(1, weight=1)
        
        l_contrasts = ttk.Label(self.random_mode_frame, text="Contrasts per Image:")
        l_contrasts.grid(row=0, column=0, sticky="w")
        e_contrasts = ttk.Entry(self.random_mode_frame, textvariable=self.num_contrasts, width=7)
        e_contrasts.grid(row=0, column=1, sticky="w", padx=5)
        Tooltip(e_contrasts, "The number of new contrast images to generate for each input image.")
        
        cb_fixed = ttk.Checkbutton(self.random_mode_frame, text="Use fixed contrasts for all images", variable=self.fixed_contrasts)
        cb_fixed.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        Tooltip(cb_fixed, "If checked, the same set of random contrast transformations\nwill be applied to every input image.")

        # Custom Mode Frame
        self.custom_mode_frame = ttk.LabelFrame(main_frame, text="Custom Mode Options", padding="10")
        self.custom_mode_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        self.custom_mode_frame.columnconfigure(0, weight=1)
        
        ttk.Button(self.custom_mode_frame, text="Create / Edit Custom Contrasts...", command=self.open_histogram_editor).grid(row=0, column=0, sticky="ew")
        self.custom_status_label = ttk.Label(self.custom_mode_frame, text="0 custom contrasts defined.")
        self.custom_status_label.grid(row=1, column=0, sticky="ew", pady=5)

        # Visual Mode Frame 
        self.visual_mode_frame = ttk.LabelFrame(main_frame, text="Visual Mode Options", padding="10")
        self.visual_mode_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        self.visual_mode_frame.columnconfigure(0, weight=1)
        
        ttk.Button(self.visual_mode_frame, text="Open Visual Editor...", command=self.open_visual_editor).grid(row=0, column=0, sticky="ew")
        self.visual_status_label = ttk.Label(self.visual_mode_frame, text="0 visually defined contrasts.")
        self.visual_status_label.grid(row=1, column=0, sticky="ew", pady=5)


        # --- ADVANCED PARAMS ---
        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Histogram Parameters", padding="10")
        adv_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=20)
        adv_frame.columnconfigure(1, weight=1)
        
        ttk.Label(adv_frame, text="Number of Bins:").grid(row=0, column=0, sticky="w", pady=2)
        e_bins = ttk.Entry(adv_frame, textvariable=self.num_bins, width=7); e_bins.grid(row=0, column=1, sticky="w", padx=5)
        Tooltip(e_bins, "Histogram resolution. Must be divisible by Chunks.")
        
        ttk.Label(adv_frame, text="Histogram Chunks:").grid(row=1, column=0, sticky="w", pady=2)
        e_chunks = ttk.Entry(adv_frame, textvariable=self.hist_chunks, width=7); e_chunks.grid(row=1, column=1, sticky="w", padx=5)
        Tooltip(e_chunks, "Number of sections to divide the histogram into for shuffling.")
        
        ttk.Label(adv_frame, text="Dark Pixel Threshold:").grid(row=2, column=0, sticky="w", pady=2)
        s_thresh = ttk.Scale(adv_frame, from_=0.0, to=1.0, orient="horizontal", variable=self.dark_threshold, command=lambda v: self.dark_thresh_label.config(text=f"{float(v):.2f}"))
        s_thresh.grid(row=2, column=1, sticky="ew", padx=5)
        self.dark_thresh_label = ttk.Label(adv_frame, text=f"{self.dark_threshold.get():.2f}")
        self.dark_thresh_label.grid(row=2, column=2, sticky="w", padx=5)
        Tooltip(s_thresh, "For Random Mode, pixels below this are not shuffled.\nFor Custom Mode, this defines the 'darkest chunk' that can optionally be included in shuffling.")

        # --- ACTION & STATUS ---
        self.generate_button = ttk.Button(main_frame, text="Generate Contrasts", command=self.start_generation_thread, style="Accent.TButton")
        self.generate_button.grid(row=7, column=0, columnspan=3, pady=20)
        self.status_label = ttk.Label(main_frame, text="Ready.", relief="sunken", anchor="w", padding=5)
        self.status_label.grid(row=8, column=0, columnspan=3, sticky="ew")
        
        self.style = ttk.Style(self); self.style.configure("Accent.TButton", font=("Helvetica", 12, "bold"))
        self.toggle_ui_mode()

    def toggle_ui_mode(self):
        mode = self.generation_mode.get()

        self.random_mode_frame.grid_remove()
        self.custom_mode_frame.grid_remove()
        self.visual_mode_frame.grid_remove()
        
        if mode == 'random':
            self.random_mode_frame.grid()
        elif mode == 'custom':
            self.custom_mode_frame.grid()
        elif mode == 'visual':
            self.visual_mode_frame.grid()

    def _validate_common_params(self):
        """Helper to validate params needed for any editor."""
        if not os.path.isdir(self.input_dir.get()):
            messagebox.showerror("Error", "Please select a valid input directory first.", parent=self)
            return False
        try:
            num_b = int(self.num_bins.get()); num_h = int(self.hist_chunks.get())
            if num_b <= 0 or num_h <= 0 or num_b % num_h != 0: raise ValueError
        except ValueError:
            messagebox.showerror("Validation Error", "Please set valid, divisible Bins and Chunks in Advanced Parameters first.", parent=self)
            return False
        return True

    def open_histogram_editor(self):
        if not self._validate_common_params(): return
        advanced_params = {
            'num_bins': int(self.num_bins.get()),
            'hist_chunks': int(self.hist_chunks.get()),
            'dark_threshold': self.dark_threshold.get()
        }
        editor = HistogramEditor(self, self.input_dir.get(), advanced_params, self.on_editor_save)
    
    def open_visual_editor(self):
        if not self._validate_common_params(): return
        advanced_params = {
            'num_bins': int(self.num_bins.get()),
            'hist_chunks': int(self.hist_chunks.get())
        }
        editor = GuidanceMapEditor(self, self.input_dir.get(), advanced_params, self.on_editor_save)


    def on_editor_save(self, permutations):
        self.defined_contrasts = permutations
        count = len(self.defined_contrasts)

        self.custom_status_label.config(text=f"{count} custom contrast(s) defined.")
        self.visual_status_label.config(text=f"{count} visually defined contrast(s).")
        self.update_status(f"Received {count} custom permutations from editor.")

    def select_input_dir(self):
        path = filedialog.askdirectory(title="Select Input Dataset Folder")
        if path:
            self.input_dir.set(path)
            if not self.output_dir.get(): self.output_dir.set(os.path.join(path, "generated_contrasts"))

    def select_output_dir(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path: self.output_dir.set(path)

    def update_status(self, message):
        self.status_label.config(text=message)

    def start_generation_thread(self):
        if not os.path.isdir(self.input_dir.get()):
            messagebox.showerror("Validation Error", "Please select a valid input directory."); return
        if not self.output_dir.get():
            messagebox.showerror("Validation Error", "Please select an output directory."); return
        
        mode = self.generation_mode.get()
        if mode in ['custom', 'visual'] and not self.defined_contrasts:
            messagebox.showerror("Validation Error", f"{mode.capitalize()} mode is selected, but no contrasts have been created."); return

        try:
            num_b = int(self.num_bins.get()); num_h = int(self.hist_chunks.get())
            if num_b % num_h != 0:
                messagebox.showerror("Validation Error", "Number of Bins must be divisible by Histogram Chunks."); return
        except ValueError:
            messagebox.showerror("Validation Error", "Bins and Chunks must be positive integers."); return

        self.generate_button.config(state="disabled")
        self.update_status("Starting...")
        
        # Determine the effective mode for the backend
        backend_mode = 'custom' if mode in ['custom', 'visual'] else 'random'

        # --- Prepare arguments for backend ---
        args = {
            "input_dir": self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "generation_mode": backend_mode,
            "num_bins": int(self.num_bins.get()),
            "hist_chunks": int(self.hist_chunks.get()),
            "dark_threshold": self.dark_threshold.get(),
            "update_callback": self.update_status,
            "num_contrasts": int(self.num_contrasts.get()),
            "fixed_contrasts": self.fixed_contrasts.get(),
            "custom_permutations": self.defined_contrasts
        }

        thread = threading.Thread(target=generate_contrasts, kwargs=args)
        thread.daemon = True
        thread.start()
        self.monitor_thread(thread)

    def monitor_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.monitor_thread(thread))
        else:
            self.generate_button.config(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()