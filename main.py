import sys
import os

# --- EXE CONSOLE FIX ---
# This must remain at the very top to prevent 'NoneType has no attribute flush' 
# errors when running as a windowed EXE without a terminal console.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import customtkinter as ctk
import yfinance as yf
import pandas as pd
import numpy as np
# TensorFlow and Sklearn imports moved inside the class to allow for a loading screen
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from PIL import Image

# Global variables for the heavy libraries to be assigned after loading
MinMaxScaler = None
Sequential = None
LSTM = None
Dense = None
Input = None
LambdaCallback = None
EarlyStopping = None

# --- PATH CONFIGURATION (Fixed for PyInstaller Compatibility) ---
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle (EXE)
    BASE_PATH = sys._MEIPASS
else:
    # If the application is run as a script
    BASE_PATH = os.path.dirname(os.path.realpath(__file__))

ASSETS_PATH = os.path.join(BASE_PATH, "assets")

# --- TOOLTIP HELPER CLASS ---
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ctk.CTkLabel(tw, text=self.text, justify="left",
                             fg_color="#334155", text_color="white",
                             corner_radius=6, padx=12, pady=8, font=("Inter", 11))
        label.pack()

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

# --- THEME SETTINGS ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SplashScreen(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Welcome to EagleFins Market Miner Professional 2026")
        
        self.geometry("600x620") 
        self.attributes("-topmost", True)
        self.overrideredirect(True) 
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (600 // 2)
        y = (screen_height // 2) - (620 // 2)
        self.geometry(f"+{x}+{y}")

        self.configure(fg_color="#0f172a")

        try:
            logo_path = os.path.join(ASSETS_PATH, "logo.png")
            scaled_logo = ctk.CTkImage(
                light_image=Image.open(logo_path),
                dark_image=Image.open(logo_path),
                size=(150, 150)
            )
            self.logo_label = ctk.CTkLabel(self, text="", image=scaled_logo)
        except Exception:
            self.logo_label = ctk.CTkLabel(self, text="📈", font=("Inter", 80))
            
        self.logo_label.pack(pady=(30, 10))
        
        self.title_label = ctk.CTkLabel(self, text="EAGLEFINS MARKET MINER PROFESSIONAL 2026", 
                                        font=("Inter", 18, "bold"), text_color="#3b82f6")
        self.title_label.pack(pady=5)

        self.disclaimer_frame = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=10)
        self.disclaimer_frame.pack(fill="both", expand=True, padx=40, pady=15)
        
        disclaimer_text = (
            "FINANCIAL DISCLAIMER:\n\n"
            "This application is for educational purposes only. "
            "Financial data is sourced from Yahoo Finance (yfinance API).\n\n"
            "Stock forecasting is speculative and involves high risk. "
            "AI predictions are based on historical trends and cannot account for "
            "sudden market shifts. NEVER invest money you cannot afford to lose."
        )
        
        self.disc_msg = ctk.CTkLabel(self.disclaimer_frame, text=disclaimer_text, 
                                     font=("Inter", 12), wraplength=450, justify="center",
                                     text_color="#94a3b8")
        self.disc_msg.pack(padx=20, pady=20, expand=True)

        # Loading Section
        self.loading_label = ctk.CTkLabel(self, text="Initializing AI Neural Workspace...", font=("Inter", 12), text_color="#3b82f6")
        self.loading_label.pack(pady=(0, 5))

        self.load_bar = ctk.CTkProgressBar(self, width=400, height=10)
        self.load_bar.set(0)
        self.load_bar.pack(pady=(0, 30))

        self.enter_btn = ctk.CTkButton(self, text="I Understand - Enter Application", 
                                        command=self.close_splash, 
                                        fg_color="#3b82f6", hover_color="#2563eb",
                                        height=45, font=("Inter", 13, "bold"))
        
        # Start the background import process with daemon=True
        self.ai_loaded = False
        threading.Thread(target=self.load_ai_backend, daemon=True).start()
        threading.Thread(target=self.animate_loading, daemon=True).start()

    def load_ai_backend(self):
        """Heavy imports performed in background to keep UI alive"""
        global MinMaxScaler, Sequential, LSTM, Dense, Input, LambdaCallback, EarlyStopping
        
        from sklearn.preprocessing import MinMaxScaler as MMS
        from tensorflow.keras.models import Sequential as Seq
        from tensorflow.keras.layers import LSTM as Lst, Dense as Den, Input as Inp
        from tensorflow.keras.callbacks import LambdaCallback as LCB, EarlyStopping as ES
        
        MinMaxScaler = MMS
        Sequential = Seq
        LSTM = Lst
        Dense = Den
        Input = Inp
        LambdaCallback = LCB
        EarlyStopping = ES
        
        self.ai_loaded = True

    def animate_loading(self):
        """Simulate progress bar based on real AI loading state"""
        progress = 0
        while progress < 0.95:
            time.sleep(0.05)
            if self.ai_loaded:
                progress += 0.05 
            else:
                progress += 0.005 
            
            if progress > 0.4:
                self.after(0, lambda: self.loading_label.configure(text="Loading TensorFlow & Scikit-Learn..."))
            if progress > 0.8:
                self.after(0, lambda: self.loading_label.configure(text="Finalizing Neural Workspace..."))
                
            self.after(0, lambda p=progress: self.load_bar.set(p))

        while not self.ai_loaded:
            time.sleep(0.1)
            
        self.after(0, lambda: self.load_bar.set(1.0))
        self.after(0, lambda: self.loading_label.configure(text="System Ready"))
        self.after(500, self.show_enter_button)

    def show_enter_button(self):
        self.loading_label.pack_forget()
        self.load_bar.pack_forget()
        self.enter_btn.pack(pady=(0, 30))

    def close_splash(self):
        self.parent.deiconify() 
        self.destroy()

class StockForecasterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.withdraw() 
        
        self.title("EagleFins Market Miner Professional 2026")
        self.geometry("1200x900")

        # FIX: Ensure application closes completely in Task Manager
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        try:
            icon_path = os.path.join(ASSETS_PATH, "icon.ico")
            self.iconbitmap(icon_path)
        except Exception:
            pass
        
        self.current_forecast_df = None
        self.input_widgets = []

        self.splash = SplashScreen(self)

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkScrollableFrame(self, width=280, corner_radius=0, fg_color="#1a1a1a")
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        
        ctk.CTkLabel(self.sidebar, text="AI SETTINGS", font=("Inter", 20, "bold")).pack(pady=(30, 20))

        # Inputs with Help Text (Tooltips)
        self.ticker_entry = self.create_input("Ticker Symbol", "AAPL", 
            "Enter a stock (AAPL) or Crypto (BTC-USD) symbol from Yahoo Finance.")
        
        self.months_entry = self.create_input("Training Period (Months)", "60", 
            "How many months of historical data the AI should study.")
        
        self.horizon_entry = self.create_input("Forecast Period (Months)", "24", 
            "How many months into the future the AI should attempt to predict.")
        
        self.epochs_entry = self.create_input("AI Training Epochs", "50", 
            "Number of times the AI scans the dataset. Higher = more detail, but slower.")
        
        self.lookback_entry = self.create_input(
            "Look-back Days", "60", 
            "Upside: Larger windows let the AI see broader patterns.\nDownside: Increases noise.",
            command=self.draw_network_map
        )
        
        self.val_split_entry = self.create_input("Validation Split % (Max 50)", "20", 
            "Percentage of data kept hidden from the AI to test its accuracy.")

        # Manual Tooltips for Slider/Switch/Dropdown
        comp_lbl = ctk.CTkLabel(self.sidebar, text="Model Complexity", font=("Inter", 12), cursor="question_arrow")
        comp_lbl.pack(pady=(10, 0))
        Tooltip(comp_lbl, "Increases the number of neurons and layers in the AI's 'brain'.")
        
        self.complexity_var = ctk.IntVar(value=2)
        self.complexity_slider = ctk.CTkSlider(self.sidebar, from_=1, to=5, number_of_steps=4, 
                                               variable=self.complexity_var, command=self.draw_network_map)
        self.complexity_slider.pack(pady=5)
        self.input_widgets.append(self.complexity_slider)

        self.early_stop_var = ctk.BooleanVar(value=True)
        self.early_stop_switch = ctk.CTkSwitch(self.sidebar, text="Enable Early Stopping", variable=self.early_stop_var, font=("Inter", 12))
        self.early_stop_switch.pack(pady=(15, 5))
        Tooltip(self.early_stop_switch, "Stops training automatically if the AI stops improving, saving time.")
        self.input_widgets.append(self.early_stop_switch)

        noise_lbl = ctk.CTkLabel(self.sidebar, text="Add Market Noise?", font=("Inter", 12), cursor="question_arrow")
        noise_lbl.pack(pady=(10, 0))
        Tooltip(noise_lbl, "Adds random volatility to the forecast to simulate real-world market chaos.")
        
        self.noise_var = ctk.StringVar(value="yes")
        self.noise_dropdown = ctk.CTkComboBox(self.sidebar, values=["yes", "no"], variable=self.noise_var, width=180)
        self.noise_dropdown.pack(pady=5)
        self.input_widgets.append(self.noise_dropdown)

        self.progress_bar = ctk.CTkProgressBar(self.sidebar, width=180)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=5)

        self.predict_btn = ctk.CTkButton(self.sidebar, text="Run Forecast", command=self.start_forecast, fg_color="#3b82f6", font=("Inter", 13, "bold"))
        self.predict_btn.pack(pady=(15, 10))

        self.save_btn = ctk.CTkButton(self.sidebar, text="Save Chart", command=self.save_chart, fg_color="#475569")
        self.save_btn.pack(pady=10)
        
        self.csv_btn = ctk.CTkButton(self.sidebar, text="Download CSV", command=self.download_csv, fg_color="#059669")
        self.csv_btn.pack(pady=10)

        self.clear_btn = ctk.CTkButton(self.sidebar, text="Clear Data", command=self.clear_ui, fg_color="#ef4444")
        self.clear_btn.pack(pady=10)

        self.status_lbl = ctk.CTkLabel(self.sidebar, text="Ready", text_color="#94a3b8")
        self.status_lbl.pack(pady=(10, 30))

        # --- MAIN VIEW ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        self.metrics_container = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.metrics_container.pack(fill="x", pady=(0, 20))
        
        self.roi_labels = {}
        for label in ["6M ROI", "12M ROI", "24M ROI"]:
            card = ctk.CTkFrame(self.metrics_container, height=80)
            card.pack(side="left", expand=True, padx=5)
            ctk.CTkLabel(card, text=label, font=("Inter", 11), text_color="#94a3b8").pack(pady=(10, 0))
            self.roi_labels[label] = ctk.CTkLabel(card, text="--", font=("Inter", 22, "bold"))
            self.roi_labels[label].pack(pady=(0, 10))

        self.tab_view = ctk.CTkTabview(self.main_frame, fg_color="#1e293b")
        self.tab_view.pack(fill="both", expand=True)
        self.tab_view.add("Stock Forecast")
        self.tab_view.add("Training Metrics")
        self.tab_view.add("Model Architecture")

        self.draw_network_map()

    def on_closing(self):
        """Ensures all background threads and the process terminate on close"""
        self.destroy()
        os._exit(0)

    def create_input(self, label, default, tip_text=None, command=None):
        lbl = ctk.CTkLabel(self.sidebar, text=label, font=("Inter", 12), cursor="question_arrow")
        lbl.pack(pady=(10, 0))
        if tip_text:
            Tooltip(lbl, tip_text)
            
        entry = ctk.CTkEntry(self.sidebar, width=180)
        entry.insert(0, default)
        entry.pack(pady=5)
        
        if command:
            entry.bind("<KeyRelease>", command)
            
        self.input_widgets.append(entry)
        return entry

    def toggle_inputs(self, enabled=True):
        state = "normal" if enabled else "disabled"
        for widget in self.input_widgets:
            widget.configure(state=state)
        self.clear_btn.configure(state=state)

    def draw_network_map(self, event=None):
        for widget in self.tab_view.tab("Model Architecture").winfo_children():
            widget.destroy()

        complexity = self.complexity_var.get()
        num_layers = max(1, complexity - 1)
        neuron_count = 32 + (complexity * 16)
        
        lookback_raw = self.lookback_entry.get()
        lookback = lookback_raw if lookback_raw != "" else "..."

        arch_container = ctk.CTkFrame(self.tab_view.tab("Model Architecture"), fg_color="transparent")
        arch_container.pack(fill="both", expand=True, padx=10, pady=10)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e293b')
        ax.set_facecolor('#1e293b')
        
        layer_names = ["Input Layer"] + [f"LSTM Layer {i+1}" for i in range(num_layers)] + ["Dense Output"]
        
        for i, name in enumerate(layer_names):
            color = '#3b82f6' if "LSTM" in name else ('#4ade80' if "Output" in name else '#94a3b8')
            rect = plt.Rectangle((i-0.4, 0.2), 0.8, 0.6, color=color, alpha=0.3)
            ax.add_patch(rect)
            plt.text(i, 0.5, f"{name}\n({neuron_count} units)" if "LSTM" in name else name, 
                     ha='center', va='center', color='white', fontsize=8, fontweight='bold')
            if i < len(layer_names) - 1:
                ax.annotate('', xy=(i+0.6, 0.5), xytext=(i+0.4, 0.5),
                            arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

        ax.set_xlim(-1, len(layer_names))
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, master=arch_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x")

        summary_frame = ctk.CTkFrame(arch_container, fg_color="#0f172a", corner_radius=10)
        summary_frame.pack(fill="both", expand=True, pady=(10, 0))

        ctk.CTkLabel(summary_frame, text="MODEL SUMMARY & DATA FLOW", font=("Inter", 14, "bold"), text_color="#3b82f6").pack(pady=(10, 5))

        summary_text = (
            f"1. DATA ENTRY: The model takes a sequence of the last {lookback} days of stock prices.\n\n"
            f"2. PATTERN RECOGNITION: The data flows into {num_layers} LSTM layer(s) with {neuron_count} neurons each.\n\n"
            f"3. FINAL PREDICTION: A single 'Dense' neuron performs the final calculation to produce the forecast."
        )

        summary_box = ctk.CTkTextbox(summary_frame, font=("Inter", 12), fg_color="transparent", wrap="word")
        summary_box.insert("0.0", summary_text)
        summary_box.configure(state="disabled")
        summary_box.pack(fill="both", expand=True, padx=15, pady=10)

    def clear_ui(self):
        for tab in ["Stock Forecast", "Training Metrics"]:
            for widget in self.tab_view.tab(tab).winfo_children():
                widget.destroy()
        for label in self.roi_labels.values():
            label.configure(text="--", text_color="#94a3b8")
        self.status_lbl.configure(text="Ready", text_color="#94a3b8")
        self.progress_bar.set(0)
        self.current_forecast_df = None

    def start_forecast(self):
        try:
            val_split = float(self.val_split_entry.get())
            if val_split > 50:
                self.status_lbl.configure(text="Validation cannot exceed 50%", text_color="#f87171")
                return
        except ValueError:
            self.status_lbl.configure(text="Invalid Validation Split", text_color="#f87171")
            return

        self.status_lbl.configure(text="AI is training...", text_color="#3b82f6")
        self.predict_btn.configure(state="disabled")
        self.toggle_inputs(False) 
        self.progress_bar.set(0)
        threading.Thread(target=self.process_logic, daemon=True).start()

    def process_logic(self):
        try:
            ticker = self.ticker_entry.get().upper()
            total_epochs = int(self.epochs_entry.get())
            val_split_dec = float(self.val_split_entry.get()) / 100
            complexity = self.complexity_var.get()
            neuron_count = 32 + (complexity * 16) 

            params = {
                "epochs": total_epochs,
                "lookback": int(self.lookback_entry.get()),
                "horizon": int(self.horizon_entry.get()),
                "add_noise": self.noise_var.get() == "yes"
            }

            # Robust Data Download for yfinance
            df = yf.download(ticker, period=f"{max(1, round(int(self.months_entry.get())/12))}y")
            if df.empty: raise Exception("No data found")
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            
            close_prices = df[['Close']].values

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            X, y = [], []
            for i in range(params['lookback'], len(scaled_data)):
                X.append(scaled_data[i-params['lookback']:i, 0])
                y.append(scaled_data[i, 0])
            
            X = np.array(X).reshape(-1, params['lookback'], 1)
            y = np.array(y)

            callbacks = [LambdaCallback(on_epoch_end=lambda epoch, logs: 
                        self.after(0, lambda: self.progress_bar.set((epoch + 1) / total_epochs)))]
            
            if self.early_stop_var.get():
                callbacks.append(EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0))

            model = Sequential()
            model.add(Input(shape=(params['lookback'], 1)))
            num_layers = max(1, complexity - 1)
            for i in range(num_layers):
                return_seq = True if i < num_layers - 1 else False
                model.add(LSTM(neuron_count, return_sequences=return_seq))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            history = model.fit(X, y, epochs=params['epochs'], batch_size=32, validation_split=val_split_dec, verbose=0, callbacks=callbacks)

            current_batch = scaled_data[-params['lookback']:].reshape((1, params['lookback'], 1))
            forecast_scaled = []
            for _ in range(params['horizon']):
                pred = model.predict(current_batch, verbose=0)[0]
                if params['add_noise']: pred = pred + np.random.normal(0, 0.012)
                forecast_scaled.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

            forecast_prices = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
            last_price = float(df['Close'].iloc[-1])
            last_date = df.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_prices), freq='ME')
            self.current_forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Close': forecast_prices})

            rois = {}
            for label, months in {"6M ROI": 6, "12M ROI": 12, "24M ROI": 24}.items():
                if len(forecast_prices) >= months:
                    val = ((forecast_prices[months-1] - last_price) / last_price) * 100
                    rois[label] = f"{val:.2f}%"
                else: rois[label] = "--"

            self.after(0, lambda: self.update_ui(df, forecast_prices, rois, history.history))
        except Exception as e:
            self.after(0, lambda: self.status_lbl.configure(text=f"Error: {str(e)}", text_color="#f87171"))
            self.after(0, lambda: self.toggle_inputs(True)) 
            self.predict_btn.configure(state="normal")

    def update_ui(self, df, forecast, rois, history_dict):
        actual_epochs = len(history_dict['loss'])
        total_epochs = int(self.epochs_entry.get())
        status_msg = "Complete!" if actual_epochs == total_epochs else f"Stopped early at {actual_epochs} epochs"
        
        self.status_lbl.configure(text=status_msg, text_color="#4ade80")
        self.predict_btn.configure(state="normal")
        self.toggle_inputs(True) 
        self.progress_bar.set(1)
        
        for k, v in rois.items():
            color = "#94a3b8" if v == "--" else ("#f87171" if "-" in v else "#4ade80")
            self.roi_labels[k].configure(text=v, text_color=color)

        for tab in ["Stock Forecast", "Training Metrics"]:
            for widget in self.tab_view.tab(tab).winfo_children(): widget.destroy()

        history_df = df.tail(150)
        history_dates = history_df.index
        forecast_dates = pd.date_range(start=history_dates[-1] + pd.Timedelta(days=1), periods=len(forecast), freq='ME')
        
        self.current_fig, ax1 = plt.subplots(figsize=(8, 5), facecolor='#1e293b')
        ax1.set_facecolor('#1e293b')
        ax1.plot(history_dates, history_df['Close'], color='#3b82f6', label="Historical Price")
        ax1.plot([history_dates[-1]] + list(forecast_dates), [history_df['Close'].iloc[-1]] + list(forecast), color='#4ade80', linestyle='--', label="AI Forecast", linewidth=2.5)
        self.current_fig.autofmt_xdate()
        ax1.tick_params(colors='white')
        ax1.legend(facecolor='#1e293b', labelcolor='white')
        
        canvas1 = FigureCanvasTkAgg(self.current_fig, master=self.tab_view.tab("Stock Forecast"))
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)

        metrics_tab = self.tab_view.tab("Training Metrics")
        fig_loss, ax2 = plt.subplots(figsize=(8, 4), facecolor='#1e293b')
        ax2.set_facecolor('#1e293b')
        ax2.plot(history_dict['loss'], color='#f87171', label="Training Loss")
        if 'val_loss' in history_dict:
            ax2.plot(history_dict['val_loss'], color='#3b82f6', label="Validation Loss")
        ax2.set_title("Model Learning Performance", color="white", pad=15)
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#1e293b', labelcolor='white')

        canvas2 = FigureCanvasTkAgg(fig_loss, master=metrics_tab)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="x", expand=False)

        loss_summary_frame = ctk.CTkFrame(metrics_tab, fg_color="#0f172a", corner_radius=10)
        loss_summary_frame.pack(fill="both", expand=True, pady=(10, 0), padx=10)

        ctk.CTkLabel(loss_summary_frame, text="METRIC INTERPRETATION", font=("Inter", 14, "bold"), text_color="#f87171").pack(pady=(10, 5))

        loss_text = (
            "1. TRAINING LOSS (Red): Measures how well the AI is fitting the training data.\n\n"
            "2. VALIDATION LOSS (Blue): Measures performance on unseen data.\n\n"
            "3. OVERFITTING WARNING: If Blue rises while Red falls, the AI is memorizing noise."
        )

        loss_box = ctk.CTkTextbox(loss_summary_frame, font=("Inter", 12), fg_color="transparent", wrap="word", height=120)
        loss_box.insert("0.0", loss_text)
        loss_box.configure(state="disabled")
        loss_box.pack(fill="both", expand=True, padx=15, pady=10)

    def save_chart(self):
        if not hasattr(self, 'current_fig') or self.current_fig is None:
            self.status_lbl.configure(text="No chart to save. Run forecast first.", text_color="#f87171")
            return
            
        file_path = ctk.filedialog.asksaveasfilename(
            parent=self,
            title="Export Forecast Chart",
            defaultextension=".png", 
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            initialfile=f"Forecast_{self.ticker_entry.get().upper()}.png"
        )
        
        if file_path:
            try:
                self.current_fig.savefig(file_path, facecolor='#1e293b', bbox_inches='tight', dpi=300)
                self.status_lbl.configure(text="Chart saved successfully!", text_color="#4ade80")
            except Exception as e:
                self.status_lbl.configure(text=f"Save Error: {str(e)}", text_color="#f87171")
        
    def download_csv(self):
        if self.current_forecast_df is None:
            self.status_lbl.configure(text="No data to export. Run forecast first.", text_color="#f87171")
            return
            
        file_path = ctk.filedialog.asksaveasfilename(
            parent=self,
            title="Export Forecast CSV",
            defaultextension=".csv", 
            filetypes=[("CSV File", "*.csv"), ("All Files", "*.*")],
            initialfile=f"Forecast_{self.ticker_entry.get().upper()}.csv"
        )
        
        if file_path:
            try:
                self.current_forecast_df.to_csv(file_path, index=False)
                self.status_lbl.configure(text="CSV data exported!", text_color="#4ade80")
            except Exception as e:
                self.status_lbl.configure(text=f"Export Error: {str(e)}", text_color="#f87171")

if __name__ == "__main__":
    app = StockForecasterApp()
    app.mainloop()