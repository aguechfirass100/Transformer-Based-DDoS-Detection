import os
import queue
import re
import shutil
import subprocess
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk


class TikTokVideoProcessor:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_styles()
        self.create_widgets()
        self.input_path = None
        self.output_path = None
        self.total_duration = None
        self.processing = False
        self.progress_queue = queue.Queue()
        self.log_file = "ffmpeg_log.txt"

    def setup_window(self):
        self.root.title("TikTok Video Processor")
        self.root.geometry("600x400")
        self.root.configure(bg="#0d1117")

    def create_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.colors = {
            "bg": "#0d1117",
            "surface": "#161b22",
            "text": "#f0f6fc",
            "accent": "#238636",
            "accent_hover": "#2ea043",
            "info": "#1f6feb",
        }
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure(
            "TLabel",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Arial", 10),
        )
        self.style.configure(
            "Title.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Arial", 20, "bold"),
        )
        self.style.configure(
            "Subtitle.TLabel",
            background=self.colors["bg"],
            foreground="#8b949e",
            font=("Arial", 8),
        )

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title_label = ttk.Label(
            main_frame, text="TikTok Video Processor", style="Title.TLabel"
        )
        title_label.pack(pady=(0, 10))

        desc_label = ttk.Label(
            main_frame,
            text="Optimize your videos for TikTok with high quality",
            style="TLabel",
        )
        desc_label.pack(pady=(0, 20))

        input_frame = ttk.Frame(main_frame, style="TFrame")
        input_frame.pack(fill="x", pady=(0, 10))

        input_label = ttk.Label(input_frame, text="Input Video:", style="TLabel")
        input_label.pack(side="left")

        self.input_path_label = ttk.Label(
            input_frame, text="No file selected", style="TLabel"
        )
        self.input_path_label.pack(side="left", padx=10)

        browse_button = tk.Button(
            input_frame,
            text="Browse",
            command=self.select_file,
            bg=self.colors["accent"],
            fg="white",
            activebackground=self.colors["accent_hover"],
            relief="flat",
            borderwidth=0,
            cursor="hand2",
        )
        browse_button.pack(side="right")
        browse_button.bind("<Enter>", self.on_button_enter)
        browse_button.bind("<Leave>", self.on_button_leave)

        output_frame = ttk.Frame(main_frame, style="TFrame")
        output_frame.pack(fill="x", pady=(0, 20))

        output_label = ttk.Label(output_frame, text="Output Video:", style="TLabel")
        output_label.pack(side="left")

        self.output_path_label = ttk.Label(output_frame, text="", style="TLabel")
        self.output_path_label.pack(side="left", padx=10)

        self.process_button = tk.Button(
            main_frame,
            text="Process Video",
            command=self.process_video,
            bg=self.colors["accent"],
            fg="white",
            activebackground=self.colors["accent_hover"],
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            state="disabled",
        )
        self.process_button.pack(pady=(0, 20))
        self.process_button.bind("<Enter>", self.on_button_enter)
        self.process_button.bind("<Leave>", self.on_button_leave)

        self.progress = ttk.Progressbar(main_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 10))

        self.status_label = ttk.Label(
            main_frame, text="Select a video file to process", style="TLabel"
        )
        self.status_label.pack()

        watermark_frame = ttk.Frame(main_frame, style="TFrame")
        watermark_frame.pack(side="bottom", fill="x", pady=(20, 0))

        watermark_label = tk.Label(
            watermark_frame,
            text="Created by 3AGGECH",
            font=("Arial", 10, "underline"),
            fg=self.colors["info"],
            bg=self.colors["bg"],
            cursor="hand2",
        )
        watermark_label.pack(side="right")
        watermark_label.bind(
            "<Button-1>", lambda e: webbrowser.open("https://t.me/3AGGECH")
        )
        watermark_label.bind(
            "<Enter>", lambda e: watermark_label.config(fg=self.colors["accent"])
        )
        watermark_label.bind(
            "<Leave>", lambda e: watermark_label.config(fg=self.colors["info"])
        )

    def on_button_enter(self, event):
        event.widget.config(bg=self.colors["accent_hover"])

    def on_button_leave(self, event):
        event.widget.config(bg=self.colors["accent"])

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mov")]
        )
        if file_path:
            self.input_path = file_path
            self.input_path_label.config(text=os.path.basename(file_path))
            base, ext = os.path.splitext(file_path)
            self.output_path = f"{base}_tiktok{ext}"
            self.output_path_label.config(text=os.path.basename(self.output_path))
            self.process_button.config(state="normal")
            self.status_label.config(text="Ready to process")

    def process_video(self):
        if not self.input_path or not self.output_path:
            messagebox.showerror("Error", "No file selected")
            return
        if self.processing:
            messagebox.showinfo("Info", "Processing is already in progress")
            return
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            messagebox.showerror(
                "Error",
                "FFmpeg or ffprobe is not installed or not in PATH. Download from https://ffmpeg.org/download.html",
            )
            return
        self.processing = True
        self.process_button.config(state="disabled")
        self.status_label.config(text="Validating video...")
        self.root.update()
        try:
            # Validate input video
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    self.input_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ValueError(f"Invalid video file: {result.stderr}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to validate video: {e}")
            self.processing = False
            self.process_button.config(state="normal")
            return
        self.status_label.config(text="Getting video duration...")
        self.root.update()
        try:
            duration_str = (
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        self.input_path,
                    ]
                )
                .decode("utf-8")
                .strip()
            )
            self.total_duration = float(duration_str)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get video duration: {e}")
            self.processing = False
            self.process_button.config(state="normal")
            return
        self.status_label.config(text="Processing video...")
        self.progress["value"] = 0
        threading.Thread(target=self.run_ffmpeg, daemon=True).start()
        self.check_progress_queue()

    def run_ffmpeg(self):
        try:
            with open(self.log_file, "w") as log:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    self.input_path,
                    "-vf",
                    "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
                    "-c:v",
                    "libx264",
                    "-b:v",
                    "15M",
                    "-minrate",
                    "15M",
                    "-maxrate",
                    "15M",
                    "-bufsize",
                    "15M",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-y",
                    self.output_path,
                ]
                proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                time_pattern = re.compile(r"time=(\d+:\d+:\d+\.\d+)")
                for line in iter(proc.stdout.readline, ""):
                    log.write(line + "\n")
                    match = time_pattern.search(line)
                    if match:
                        time_str = match.group(1)
                        current_time = self.parse_time(time_str)
                        if self.total_duration > 0:
                            progress = (current_time / self.total_duration) * 100
                            self.progress_queue.put(progress)
                proc.wait()
                if proc.returncode == 0:
                    self.progress_queue.put("done")
                else:
                    self.progress_queue.put("error: FFmpeg process failed")
        except Exception as e:
            self.progress_queue.put(f"error: {str(e)}")

    def parse_time(self, time_str):
        try:
            parts = time_str.split(":")
            if len(parts) != 3:
                raise ValueError(f"Unexpected time format: {time_str}")
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception as e:
            with open(self.log_file, "a") as log:
                log.write(f"Error parsing time '{time_str}': {e}\n")
            return 0  # Fallback to avoid crashing, progress may not update accurately

    def check_progress_queue(self):
        try:
            while True:
                item = self.progress_queue.get_nowait()
                if isinstance(item, float):
                    self.progress["value"] = item
                elif item == "done":
                    self.progress["value"] = 100
                    messagebox.showinfo(
                        "Success",
                        f"Video processed successfully: {os.path.basename(self.output_path)}",
                    )
                    self.status_label.config(text="Processing complete")
                    self.processing = False
                    break
                elif item.startswith("error"):
                    messagebox.showerror(
                        "Error",
                        f"Failed to process video: {item}. Check ffmpeg_log.txt for details.",
                    )
                    self.status_label.config(text="Processing failed")
                    self.processing = False
                    break
        except queue.Empty:
            pass
        if self.processing:
            self.root.after(100, self.check_progress_queue)
        else:
            self.process_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = TikTokVideoProcessor(root)
    root.mainloop()
