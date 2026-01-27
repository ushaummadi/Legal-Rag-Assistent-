import subprocess
import sys

def main():
    subprocess.check_call([sys.executable, "-m", "src.ingestion.run_ingestion"])
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py"])

if __name__ == "__main__":
    main()
