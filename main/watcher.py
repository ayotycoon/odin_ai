import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class StreamlitReloader(FileSystemEventHandler):
    def __init__(self, script):
        self.script = script
        self.process = None
        self.start_streamlit()

    def start_streamlit(self):
        if self.process:
            self.process.kill()
        self.process = subprocess.Popen(["streamlit", "run", self.script])


    def on_modified(self, event):
        if event.src_path.endswith("streamlit_main.py"):  # Restart only on .py file changes
            print(f"File {event.src_path} changed. Restarting Streamlit...")
            self.start_streamlit()

def watch(directory, script):
    event_handler = StreamlitReloader(script)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    watch("main/GUI", "main/GUI/streamlit_main.py")  # Update the path to your main Streamlit script
