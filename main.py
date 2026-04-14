import subprocess
import time

subprocess.Popen(["python", "utils/triggers/trigger_server.py"])
time.sleep(2)
# subprocess.Popen(["python", "utils/triggers/trigger_client.py"])