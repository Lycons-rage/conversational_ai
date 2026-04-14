import subprocess
import time

server_proc = subprocess.Popen(["python", "utils/triggers/trigger_server.py"])
time.sleep(2)
# subprocess.Popen(["python", "utils/triggers/trigger_client.py"])

try:
    server_proc.wait()
except KeyboardInterrupt:
    print("Killing server...")
    server_proc.terminate()
    server_proc.wait()