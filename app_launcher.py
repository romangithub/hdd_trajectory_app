# app_launcher.py — прозрачный запуск Streamlit приложения как «программы»
import os
import sys
from pathlib import Path

def main():
    # Переключим рабочую директорию на папку с приложением (рядом с Home.py)
    app_dir = Path(__file__).resolve().parent
    os.chdir(app_dir)

    # Аргументы запуска streamlit
    # --server.runOnSave=false: чтобы в сборке не делать hot-reload
    # --browser.gatherUsageStats=false: чтобы не спрашивал
    argv = [
        "streamlit",
        "run",
        "Home.py",
        "--server.headless=false",
        "--server.runOnSave=false",
        "--browser.gatherUsageStats=false",
    ]

    # Streamlit bootstrap (официальный способ программного запуска)
    try:
        from streamlit.web import bootstrap
    except Exception:
        # fallback для старых версий
        from streamlit import bootstrap
    bootstrap.run(file="Home.py", command_line=argv)

if __name__ == "__main__":
    # Защитим от двойного запуска на Windows
    if getattr(sys, "frozen", False):
        # В режиме сборки PyInstaller
        pass
    main()
