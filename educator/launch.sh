echo "Don't forget to create 'token.txt' with your Pinggy API token before running this script."
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
python educator.py