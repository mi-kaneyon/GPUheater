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
        self.root.title("GPU Load Tool")
        self.style = ttk.Style()
        self.create_widgets()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.update_gpu_stats, daemon=True)
        self.monitor_thread.start()
        self.load_thread = None
        self.running_load = False
        
        # --- MODIFICATION 1: VRAMを保持するためのリスト ---
        # This list will hold onto tensors to keep VRAM occupied.
        self.vram_hog_list = []

    def create_widgets(self):
        ttk.Label(self.root, text="Load Level:").pack(pady=5)
        self.load_scale = ttk.Scale(self.root, from_=0, to=100, orient=HORIZONTAL, length=300, command=self.update_load)
        self.load_scale.pack(pady=5)
        ttk.Label(self.root, text="GPU Load").pack()
        self.gpu_load_meter = ttk.Meter(self.root, metersize=180, padding=5, amounttotal=100, metertype='semi', subtext='%', bootstyle='primary')
        self.gpu_load_meter.pack(pady=5)
        ttk.Label(self.root, text="FAN Speed").pack()
        self.fan_speed_meter = ttk.Meter(self.root, metersize=180, padding=5, amounttotal=100, metertype='semi', subtext='%', bootstyle='success')
        self.fan_speed_meter.pack(pady=5)
        ttk.Label(self.root, text="VRAM Usage").pack()
        self.vram_usage_meter = ttk.Meter(self.root, metersize=180, padding=5, amounttotal=100, metertype='semi', subtext='%', bootstyle='info')
        self.vram_usage_meter.pack(pady=5)

    def update_load(self, value):
        load_level = int(float(value))
        if load_level > 0 and not self.running_load:
            self.running_load = True
            self.load_thread = threading.Thread(target=self.generate_load, daemon=True)
            self.load_thread.start()
        elif load_level == 0 and self.running_load:
            self.running_load = False
            if self.load_thread:
                self.load_thread.join(0.1)

    def generate_load(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cpu':
            print("CUDA is not available.")
            self.running_load = False
            return

        print("Load generation started.")
        while self.running_load:
            try:
                current_load = int(self.load_scale.get())
                if current_load == 0:
                    break

                # --- MODIFICATION 2: VRAM使用量の制御 ---
                # スライダーの値に基づいて、保持するテンソルの数を決定
                # 4070(12GB)に合わせて調整。メモリ1GBあたり約1つのテンソルを確保
                num_tensors_to_hog = int(current_load / 100 * 10) # Max 10GB VRAM
                
                # VRAMを増やす
                while len(self.vram_hog_list) < num_tensors_to_hog:
                    # 約1GBのテンソルを確保
                    tensor_hog = torch.randn(1, 256, 1024, 1024, device=device)
                    self.vram_hog_list.append(tensor_hog)
                    print(f"Allocating VRAM... Current hog: {len(self.vram_hog_list)} GB")

                # VRAMを減らす
                while len(self.vram_hog_list) > num_tensors_to_hog:
                    self.vram_hog_list.pop()
                    print(f"Releasing VRAM... Current hog: {len(self.vram_hog_list)} GB")

                # --- MODIFICATION 3: GPU負荷の細かな制御 ---
                # 負荷レベルに応じて計算量を増減
                tensor_size = 500 + current_load * 40
                a = torch.randn(tensor_size, tensor_size, device=device)
                b = torch.randn(tensor_size, tensor_size, device=device)
                c = torch.matmul(a, b)
                
                # 負荷レベルに応じてスリープ時間を調整し、平均負荷を制御
                # 負荷100の時はほぼスリープせず、負荷が低いほど長くスリープする
                sleep_duration = (100 - current_load) / 2000.0
                time.sleep(sleep_duration)

            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory! Please reduce the load or adjust the code.")
                self.vram_hog_list.pop() # メモリ不足なので一つ解放する
                time.sleep(2) # 少し待つ
            except Exception as e:
                print(f"An error occurred during load generation: {e}")
                break
        
        # 負荷が0になったら確保したVRAMをすべて解放
        self.vram_hog_list.clear()
        torch.cuda.empty_cache()
        print("Load generation stopped and VRAM cleared.")
        self.running_load = False

    def update_gpu_stats(self):
        # (This function remains unchanged)
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            while self.monitoring:
                utilization = nvmlDeviceGetUtilizationRates(handle)
                self.gpu_load_meter.configure(amountused=utilization.gpu)
                try:
                    fan_speed = nvmlDeviceGetFanSpeed(handle)
                    self.fan_speed_meter.configure(amountused=fan_speed)
                except NVMLError:
                    self.fan_speed_meter.configure(amountused=0)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                vram_usage = (mem_info.used / mem_info.total) * 100
                self.vram_usage_meter.configure(amountused=vram_usage)
                time.sleep(1)
        except NVMLError as error:
            print(f"Failed to communicate with NVIDIA driver: {error}")
        finally:
            try: nvmlShutdown()
            except: pass

    def on_closing(self):
        self.monitoring = False
        self.running_load = False
        if self.load_thread and self.load_thread.is_alive():
            self.load_thread.join()
        self.root.destroy()


if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    app = GpuLoadTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
