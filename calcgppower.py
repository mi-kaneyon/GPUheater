import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import torch
import threading
import time
from pynvml import *

class GpuLoadTool:
    def __init__(self, root):
        self.root = root
        self.root.title("CUDA Stress Test Tool (Multi-Mode)")
        self.style = ttk.Style()
        
        # Current status flags
        self.monitoring = True
        self.running_load = False
        self.load_thread = None
        
        # List to hold tensors for VRAM occupation
        self.vram_hog_list = []

        # --- FIX: Create widgets FIRST so they exist when the thread runs ---
        self.create_widgets()

        # --- Start monitoring thread AFTER creating widgets to avoid AttributeError ---
        self.monitor_thread = threading.Thread(target=self.update_gpu_stats, daemon=True)
        self.monitor_thread.start()

    def create_widgets(self):
        # --- Stress Mode Selection ---
        ttk.Label(self.root, text="Stress Mode:", bootstyle="inverse-secondary").pack(pady=(10, 0))
        self.mode_var = tk.StringVar(value="Matrix Multiplication")
        self.mode_combo = ttk.Combobox(self.root, textvariable=self.mode_var, state="readonly", width=30)
        self.mode_combo['values'] = (
            "Matrix Multiplication (Balanced)", 
            "Heavy Math (ALU Intensive)", 
            "VRAM Bandwidth (Memory Copy)"
        )
        self.mode_combo.pack(pady=5)

        # --- Load Slider ---
        ttk.Label(self.root, text="Load Intensity:").pack(pady=5)
        self.load_scale = ttk.Scale(self.root, from_=0, to=100, orient=HORIZONTAL, length=300, command=self.update_load)
        self.load_scale.pack(pady=5)

        # --- Meters ---
        ttk.Label(self.root, text="GPU Core Load").pack()
        self.gpu_load_meter = ttk.Meter(self.root, metersize=160, padding=5, amounttotal=100, metertype='semi', subtext='%', bootstyle='primary')
        self.gpu_load_meter.pack(pady=5)

        ttk.Label(self.root, text="FAN Speed").pack()
        self.fan_speed_meter = ttk.Meter(self.root, metersize=160, padding=5, amounttotal=100, metertype='semi', subtext='%', bootstyle='success')
        self.fan_speed_meter.pack(pady=5)

        ttk.Label(self.root, text="VRAM Usage").pack()
        self.vram_usage_meter = ttk.Meter(self.root, metersize=160, padding=5, amounttotal=100, metertype='semi', subtext='%', bootstyle='info')
        self.vram_usage_meter.pack(pady=5)

    def update_load(self, value):
        """Callback for the slider interaction."""
        load_level = int(float(value))
        
        # Start the thread if load > 0 and not already running
        if load_level > 0 and not self.running_load:
            self.running_load = True
            self.load_thread = threading.Thread(target=self.generate_load, daemon=True)
            self.load_thread.start()
        
        # Stop the logic if load is 0
        elif load_level == 0 and self.running_load:
            self.running_load = False

    def generate_load(self):
        """Main loop for generating CUDA load."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cpu':
            print("CUDA unavailable. Aborting.")
            self.running_load = False
            return

        print(f"Load generation started. Mode: {self.mode_var.get()}")

        # Pre-allocate reuseable tensors to avoid allocation overhead inside the loop (optional optimization)
        
        while self.running_load:
            try:
                current_load = int(self.load_scale.get())
                if current_load == 0:
                    break

                # --- 1. VRAM Occupation Logic ---
                # Occupy VRAM based on load percentage (approx 10GB max for reference)
                target_hog_count = int(current_load / 100 * 10)
                current_hog_count = len(self.vram_hog_list)

                if current_hog_count < target_hog_count:
                    try:
                        # Allocate ~1GB chunk
                        t = torch.randn(1, 256, 1024, 1024, device=device)
                        self.vram_hog_list.append(t)
                    except torch.cuda.OutOfMemoryError:
                        pass # Ignore if full
                elif current_hog_count > target_hog_count:
                    self.vram_hog_list.pop()

                # --- 2. Stress Computation Logic ---
                # Select operation based on the combobox
                mode = self.mode_var.get()
                
                # Base size scales with slider
                # Base size: 1000 to ~5000
                size = 1000 + (current_load * 40)

                if "Matrix" in mode:
                    # [Balanced Heat] Standard MatMul
                    a = torch.randn(size, size, device=device)
                    b = torch.randn(size, size, device=device)
                    # Matrix multiplication is very optimized and heats up Tensor Cores/CUDA cores
                    c = torch.matmul(a, b)

                elif "Heavy Math" in mode:
                    # [ALU Intensive] Trigonometry and Exponentials
                    # This stresses the FP32 units differently than MatMul
                    a = torch.randn(size, size, device=device)
                    # Complex operations chain
                    c = torch.sin(a) * torch.cos(a) + torch.exp(torch.abs(a) * 0.01)

                elif "Bandwidth" in mode:
                    # [Memory Controller Heat] Copy operations
                    # Create a large tensor and copy it repeatedly
                    # Making size slightly larger for memory tasks
                    mem_size = size + 1000 
                    a = torch.randn(mem_size, mem_size, device=device)
                    b = torch.empty_like(a)
                    b.copy_(a) # Explicit copy
                    c = b + a  # Simple add to prevent optimization removal

                # --- 3. Synchronization & Sleep ---
                # Important: Wait for GPU to finish commands before sleeping
                torch.cuda.synchronize()

                # Sleep control to throttle load (Inverse of load level)
                if current_load < 100:
                    time.sleep((100 - current_load) / 2000.0)
                else:
                    # At 100%, no sleep = Maximum thermal output
                    pass

            except torch.cuda.OutOfMemoryError:
                # If we hit VRAM limit, release some memory and wait
                print("OOM Event: Releasing memory...")
                if self.vram_hog_list:
                    self.vram_hog_list.pop()
                torch.cuda.empty_cache()
                time.sleep(1)
            except Exception as e:
                print(f"Error in load loop: {e}")
                break
        
        # Cleanup
        self.vram_hog_list.clear()
        torch.cuda.empty_cache()
        print("Load generation stopped.")
        self.running_load = False

    def update_gpu_stats(self):
        """Monitor GPU stats using NVML."""
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            while self.monitoring:
                # GPU Utilization
                utilization = nvmlDeviceGetUtilizationRates(handle)
                self.gpu_load_meter.configure(amountused=utilization.gpu)
                
                # Fan Speed
                try:
                    fan_speed = nvmlDeviceGetFanSpeed(handle)
                    self.fan_speed_meter.configure(amountused=fan_speed)
                except NVMLError:
                    # Some GPUs don't report fan speed via NVML
                    self.fan_speed_meter.configure(amountused=0)
                
                # VRAM Usage
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                vram_usage = (mem_info.used / mem_info.total) * 100
                self.vram_usage_meter.configure(amountused=vram_usage)
                
                time.sleep(1)
        except NVMLError as error:
            print(f"NVML Error: {error}")
        finally:
            try: nvmlShutdown()
            except: pass

    def on_closing(self):
        self.monitoring = False
        self.running_load = False
        if self.load_thread and self.load_thread.is_alive():
            self.load_thread.join(1.0)
        self.root.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    app = GpuLoadTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
