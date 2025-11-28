import subprocess
import sys
import os

def run_streamlit():
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    run_streamlit()
