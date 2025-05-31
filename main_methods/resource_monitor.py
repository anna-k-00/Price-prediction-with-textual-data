import threading
import time
import psutil
import logging

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class ResourceMonitor:
    def __init__(self, logger=None, interval=30):
        """
        Initialize the ResourceMonitor with an optional logger and monitoring interval.
        
        Args:
            logger (logging.Logger, optional): Logger instance for logging resource usage. Defaults to None.
            interval (int, optional): Time interval (in seconds) between monitoring checks. Defaults to 30.
        """
        self.logger = logger or logging.getLogger("ResourceMonitor")
        self.interval = interval
        self._monitor_thread = None
        self._stop_event = threading.Event()

    def get_cpu_utilization(self):
        """
        Retrieve the current CPU utilization percentage.
        
        Returns:
            float: CPU usage percentage.
        """
        return psutil.cpu_percent(interval=0.3)

    def get_ram_utilization(self):
        """
        Retrieve the current RAM utilization in gigabytes.
        
        Returns:
            tuple: A tuple containing (used_ram_gb, total_ram_gb).
        """
        mem = psutil.virtual_memory()
        return mem.used / (1024 ** 3), mem.total / (1024 ** 3)

    def get_gpu_utilization(self):
        """
        Retrieve GPU utilization and memory usage if available.
        
        Returns:
            dict or None: A dictionary containing GPU usage and memory info, or None if GPU is not available.
        """
        if HAS_NVML:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'gpu_percent': util.gpu,
                'gpu_mem_used_gb': meminfo.used / (1024 ** 3),
                'gpu_mem_total_gb': meminfo.total / (1024 ** 3)
            }
        elif torch and torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return {
                'gpu_percent': None,
                'gpu_mem_used_gb': mem_used,
                'gpu_mem_total_gb': mem_total
            }
        else:
            return None

    def log_resources(self, stage=""):
        """
        Log the current resource utilization (CPU, RAM, and GPU if available).
        
        Args:
            stage (str, optional): Additional context or stage identifier for the log message. Defaults to "".
        """
        cpu = self.get_cpu_utilization()
        ram_used, ram_total = self.get_ram_utilization()
        gpu_info = self.get_gpu_utilization()
        msg = f"[{stage}] CPU: {cpu:.1f}%, RAM: {ram_used:.2f}GB/{ram_total:.2f}GB"
        if gpu_info:
            msg += (f", GPU: {gpu_info.get('gpu_percent', 'NA')}%, "
                    f"GPU RAM: {gpu_info['gpu_mem_used_gb']:.2f}/"
                    f"{gpu_info['gpu_mem_total_gb']:.2f}GB")
        self.logger.info(msg)

    def start(self):
        """
        Start the resource monitoring thread.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.warning("ResourceMonitor is already running.")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started.")

    def stop(self):
        """
        Stop the resource monitoring thread.
        """
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join()
            self.logger.info("Resource monitoring stopped.")

    def _monitor_loop(self):
        """
        Internal loop for periodically logging resource usage.
        """
        while not self._stop_event.is_set():
            self.log_resources(stage="monitor")
            time.sleep(self.interval)
