@echo off
echo Don't forget to create 'token.txt' with your Pinggy API token before running this script.
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py