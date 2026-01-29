import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
#  SECTION 1: PURE MANUAL DSP BACKEND (The Engine)
# =============================================================================

class DSP_Backend:
    def __init__(self, fs=360):
        self.fs = fs
        self.scaler = MinMaxScaler()
        self.classifier = KNeighborsClassifier(n_neighbors=5)
        self.is_trained = False

    def load_data(self, file_path):
        data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    # Parse pipe-separated values (e.g., "-0.05|-0.06|...")
                    values = [float(x) for x in line.strip().split('|') if x]
                    if values:
                        data.append(values)
            return np.array(data)
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")

    # --- 1. Manual IIR Filter Design (Bilinear Transform) ---
    def calculate_biquad_coefficients(self, filter_type, fc, Q=0.707):
        # Calculate digital frequency w0
        w0 = 2 * math.pi * fc / self.fs
        alpha = math.sin(w0) / (2 * Q)
        cos_w0 = math.cos(w0)

        b = np.zeros(3)
        a = np.zeros(3)

        if filter_type == 'lowpass':
            b[0] = (1 - cos_w0) / 2
            b[1] = 1 - cos_w0
            b[2] = (1 - cos_w0) / 2
            a[0] = 1 + alpha
            a[1] = -2 * cos_w0
            a[2] = 1 - alpha
            
        elif filter_type == 'highpass':
            b[0] = (1 + cos_w0) / 2
            b[1] = -(1 + cos_w0)
            b[2] = (1 + cos_w0) / 2
            a[0] = 1 + alpha
            a[1] = -2 * cos_w0
            a[2] = 1 - alpha

        # Normalize coefficients
        return b / a[0], a / a[0]

    # --- 2. Manual Difference Equation ---
    def apply_difference_equation(self, data_1d, b, a):
        N = len(data_1d)
        y = np.zeros(N)
        x = data_1d
        
        for n in range(N):
            # Feedforward (Inputs)
            val = b[0]*x[n]
            if n >= 1: val += b[1]*x[n-1]
            if n >= 2: val += b[2]*x[n-2]
            
            # Feedback (Previous Outputs)
            if n >= 1: val -= a[1]*y[n-1]
            if n >= 2: val -= a[2]*y[n-2]
            
            y[n] = val
        return y

    # --- 3. Zero-Phase Filtering (Forward-Backward) ---
    def apply_zero_phase_filter(self, data):
        # Design Bandpass Strategy: HighPass (0.5Hz) + LowPass (40Hz)
        b_hp, a_hp = self.calculate_biquad_coefficients('highpass', 0.5)
        b_lp, a_lp = self.calculate_biquad_coefficients('lowpass', 40.0)

        filtered_rows = []
        for row in data:
            # Stage A: Remove Baseline Wander (High Pass)
            # 1. Forward
            hp_fwd = self.apply_difference_equation(row, b_hp, a_hp)
            # 2. Backward (Reverse -> Filter -> Reverse) to cancel phase shift
            hp_back = self.apply_difference_equation(hp_fwd[::-1], b_hp, a_hp)[::-1]
            
            # Stage B: Remove High Freq Noise (Low Pass)
            # 1. Forward
            lp_fwd = self.apply_difference_equation(hp_back, b_lp, a_lp)
            # 2. Backward
            lp_back = self.apply_difference_equation(lp_fwd[::-1], b_lp, a_lp)[::-1]
            
            filtered_rows.append(lp_back)
            
        return np.array(filtered_rows)

    # --- 4. Manual Autocorrelation (Rhythm Extraction) ---
    def manual_autocorrelation(self, signal_row, lags=180):
        N = len(signal_row)
        r = []
        for k in range(lags):
            sum_val = 0.0
            for n in range(N - k):
                sum_val += signal_row[n] * signal_row[n + k]
            r.append(sum_val)
        r = np.array(r)
        # Normalize to 1 at lag 0
        return r / r[0] if r[0] != 0 else r

    # --- 5. Manual DCT (Energy Compaction) ---
    def manual_dct(self, ac_signal, num_coeffs=15):
        N = len(ac_signal)
        X = np.zeros(num_coeffs)
        for k in range(num_coeffs):
            sum_val = 0.0
            for n in range(N):
                # DCT-II Formula
                angle = (math.pi / N) * (n + 0.5) * k
                sum_val += ac_signal[n] * math.cos(angle)
            
            # Orthogonal scaling
            scale = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
            X[k] = sum_val * scale
        return X

    def process_pipeline(self, raw_data):
        # 1. Filter
        filtered = self.apply_zero_phase_filter(raw_data)
        # 2. Normalize (0 to 1)
        norm_data = self.scaler.fit_transform(filtered.T).T
        
        features = []
        for row in norm_data:
            # 3. AutoCorr
            ac = self.manual_autocorrelation(row, lags=180)
            # 4. DCT
            dct_vals = self.manual_dct(ac, num_coeffs=15)
            features.append(dct_vals)
            
        return np.array(features)

    # --- ML Logic ---
    def train_model(self, norm_path, pvc_path):
        norm_data = self.load_data(norm_path)
        pvc_data = self.load_data(pvc_path)
        # Labels: 0 = Normal, 1 = PVC
        y = np.concatenate([np.zeros(len(norm_data)), np.ones(len(pvc_data))])
        X_raw = np.concatenate([norm_data, pvc_data])
        
        X_feats = self.process_pipeline(X_raw)
        self.classifier.fit(X_feats, y)
        self.is_trained = True
        return len(norm_data), len(pvc_data)

    def test_model(self, norm_path, pvc_path):
        if not self.is_trained: raise Exception("Model not trained!")
        norm_data = self.load_data(norm_path)
        pvc_data = self.load_data(pvc_path)
        y_true = np.concatenate([np.zeros(len(norm_data)), np.ones(len(pvc_data))])
        X_raw = np.concatenate([norm_data, pvc_data])
        
        X_feats = self.process_pipeline(X_raw)
        y_pred = self.classifier.predict(X_feats)
        return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred), classification_report(y_true, y_pred, target_names=['Normal', 'PVC'])

    def predict_user_file(self, file_path):
        """
        @desc Processes a patient file with a Medical Threshold to avoid False Positives.
        """
        if not self.is_trained: raise Exception("Model not trained! Go to Tab 1 and Train first.")
        
        # 1. Load and Process
        data = self.load_data(file_path)
        features = self.process_pipeline(data)
        
        # 2. Predict
        predictions = self.classifier.predict(features)
        
        pvc_count = np.sum(predictions)      # Count of abnormal beats
        total_beats = len(predictions)       # Total beats
        pvc_ratio = (pvc_count / total_beats) * 100  # Percentage
        
        # 3. Smart Diagnosis Logic (Thresholding)
        # We only flag "POSITIVE" if more than 5% of beats are abnormal.
        # This ignores occasional noise/errors.
        threshold_percent = 5.0 
        
        if pvc_ratio > threshold_percent:
            result = "POSITIVE"
        else:
            result = "NEGATIVE"
            
        return result, pvc_count, total_beats
# =============================================================================
#  SECTION 2: GUI (User & Engineer Tabs)
# =============================================================================

class ProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CardioLogic Pro | ECG Analysis System")
        self.root.geometry("1000x800")
        self.root.configure(bg="#1e1e1e")
        self.backend = DSP_Backend()
        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#333", foreground="white", padding=[15, 8], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#007acc")])

        # Header
        tk.Label(self.root, text="CardioLogic Pro: ECG Classification System", 
                 bg="#1e1e1e", fg="#00ff00", font=("Segoe UI", 18, "bold")).pack(pady=15)

        # Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        # Tab 1: Engineering (Training)
        self.tab_dev = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_dev, text="  1. Model Development (Train/Test)  ")
        self.setup_dev_tab()

        # Tab 2: User (Diagnosis)
        self.tab_user = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_user, text="  2. Patient Diagnosis (User Input)  ")
        self.setup_user_tab()

    # --- TAB 1 SETUP ---
    def setup_dev_tab(self):
        # Controls
        controls = tk.LabelFrame(self.tab_dev, text=" Engineer Controls ", bg="#2d2d2d", fg="white", font=("Segoe UI", 11, "bold"))
        controls.pack(side="left", fill="y", padx=10, pady=10, ipady=10)

        # Variables
        self.p_tr_n = tk.StringVar()
        self.p_tr_p = tk.StringVar()
        self.p_ts_n = tk.StringVar()
        self.p_ts_p = tk.StringVar()

        # Train
        tk.Label(controls, text="Training Data", bg="#2d2d2d", fg="#4da6ff", font=("Segoe UI", 10)).pack(pady=(10, 5))
        self.add_file_row(controls, "Normal Train:", self.p_tr_n)
        self.add_file_row(controls, "PVC Train:", self.p_tr_p)
        tk.Button(controls, text="TRAIN MODEL", bg="#006600", fg="white", font=("Segoe UI", 10, "bold"), 
                  command=self.run_train).pack(fill="x", padx=10, pady=10)

        # Test
        tk.Label(controls, text="Testing Data", bg="#2d2d2d", fg="#ff4d4d", font=("Segoe UI", 10)).pack(pady=(20, 5))
        self.add_file_row(controls, "Normal Test:", self.p_ts_n)
        self.add_file_row(controls, "PVC Test:", self.p_ts_p)
        tk.Button(controls, text="TEST MODEL", bg="#990000", fg="white", font=("Segoe UI", 10, "bold"), 
                  command=self.run_test).pack(fill="x", padx=10, pady=10)

        # Viz
        tk.Label(controls, text="Analysis", bg="#2d2d2d", fg="#ffff00", font=("Segoe UI", 10)).pack(pady=(20, 5))
        tk.Button(controls, text="Visualize Pipeline Step-by-Step", bg="#cc9900", fg="black", 
                  command=self.visualize).pack(fill="x", padx=10, pady=5)

        # Logs
        log_frame = tk.LabelFrame(self.tab_dev, text=" System Logs ", bg="#2d2d2d", fg="white")
        log_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.log_area = scrolledtext.ScrolledText(log_frame, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)

    # --- TAB 2 SETUP (USER INPUT) ---
    def setup_user_tab(self):
        container = tk.Frame(self.tab_user, bg="#1e1e1e")
        container.pack(expand=True, fill="both", padx=40, pady=40)

        # Instruction
        tk.Label(container, text="Upload Patient ECG File for Arrhythmia Diagnosis", 
                 bg="#1e1e1e", fg="white", font=("Segoe UI", 16)).pack(pady=(0, 30))

        # File Selection
        input_box = tk.Frame(container, bg="#2d2d2d", padx=20, pady=20)
        input_box.pack(fill="x")
        
        self.user_file_var = tk.StringVar()
        tk.Label(input_box, text="Patient File:", bg="#2d2d2d", fg="white", font=("Segoe UI", 12)).pack(side="left")
        tk.Entry(input_box, textvariable=self.user_file_var, width=50, font=("Segoe UI", 11)).pack(side="left", padx=15)
        tk.Button(input_box, text="Browse...", bg="#555", fg="white", font=("Segoe UI", 10),
                  command=lambda: self.browse(self.user_file_var)).pack(side="left")

        # Action Button
        tk.Button(container, text="ANALYZE PATIENT", bg="#007acc", fg="white", font=("Segoe UI", 14, "bold"),
                  command=self.run_user_diagnosis, width=20, pady=10).pack(pady=40)

        # Result Display Area
        self.res_frame = tk.Frame(container, bg="black", bd=2, relief="sunken")
        self.res_frame.pack(fill="both", expand=True)

        self.lbl_result = tk.Label(self.res_frame, text="STATUS: WAITING FOR INPUT", 
                                   bg="black", fg="#666", font=("Segoe UI", 26, "bold"))
        self.lbl_result.pack(expand=True)

        self.lbl_details = tk.Label(self.res_frame, text="", bg="black", fg="#aaa", font=("Consolas", 14))
        self.lbl_details.pack(pady=20)

    # --- HELPERS ---
    def add_file_row(self, parent, text, var):
        f = tk.Frame(parent, bg="#2d2d2d")
        f.pack(fill="x", padx=5, pady=2)
        tk.Label(f, text=text, bg="#2d2d2d", fg="#ccc", width=12, anchor="w").pack(side="left")
        tk.Button(f, text="Browse", command=lambda: self.browse(var), bg="#555", fg="white", width=8).pack(side="right")
        tk.Label(f, textvariable=var, bg="#2d2d2d", fg="#888", width=15).pack(side="right", padx=5)

    def browse(self, var):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path: var.set(path)

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)

    # --- LOGIC ---
    def run_train(self):
        if not (self.p_tr_n.get() and self.p_tr_p.get()): return messagebox.showerror("Error", "Select Training Files")
        self.log("--- Training ---")
        try:
            n, p = self.backend.train_model(self.p_tr_n.get(), self.p_tr_p.get())
            self.log(f"Loaded {n} Normal, {p} PVC beats.")
            self.log("Applied Manual Biquad Filter (HP+LP) + Zero Phase.")
            self.log("Calculated Manual Autocorrelation & DCT.")
            self.log("Training Complete.")
            messagebox.showinfo("Success", "Model Trained Successfully")
        except Exception as e: self.log(f"Error: {e}")

    def run_test(self):
        if not (self.p_ts_n.get() and self.p_ts_p.get()): return messagebox.showerror("Error", "Select Testing Files")
        self.log("--- Testing ---")
        try:
            acc, cm, rep = self.backend.test_model(self.p_ts_n.get(), self.p_ts_p.get())
            self.log(f"Accuracy: {acc*100:.2f}%")
            self.log(f"Confusion Matrix:\n{cm}")
            self.log(f"Report:\n{rep}")
        except Exception as e: self.log(f"Error: {e}")

    def run_user_diagnosis(self):
        path = self.user_file_var.get()
        if not path:
            return messagebox.showerror("Error", "Please select a file to analyze.")
        
        try:
            status, pvc_count, total = self.backend.predict_user_file(path)
            
            if status == "POSITIVE":
                self.lbl_result.config(text="⚠️ POSITIVE (Arrhythmia Detected)", fg="#ff3333")
            else:
                self.lbl_result.config(text="✅ NEGATIVE (Normal Rhythm)", fg="#00ff00")
            
            self.lbl_details.config(text=f"Analyzed {total} Heartbeats\nDetected {pvc_count} PVC (Abnormal) Beats")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def visualize(self):
        path = filedialog.askopenfilename(title="Select Signal to Visualize")
        if not path: return
        try:
            raw = self.backend.load_data(path)[0] # First beat
            
            # Manual Filter Pipeline
            filtered = self.backend.apply_zero_phase_filter(np.array([raw]))[0]
            
            # Norm
            norm = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
            
            # AutoCorr
            ac = self.backend.manual_autocorrelation(norm)
            
            # DCT
            dct = self.backend.manual_dct(ac[:180], 15)
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax[0,0].plot(raw, 'r'); ax[0,0].set_title("1. Raw")
            ax[0,1].plot(norm, 'c'); ax[0,1].set_title("2. Manual Filtered & Norm")
            ax[1,0].plot(ac[:200], 'g'); ax[1,0].set_title("3. Manual AutoCorr")
            
            

            
            ax[1,1].stem(dct, linefmt='y-'); ax[1,1].set_title("4. Manual DCT Features")
            plt.tight_layout()
            plt.show()
        except Exception as e: messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectGUI(root)
    root.mainloop()