import subprocess

subprocess.run([
    "uvicorn",
    "services.server:app",
    "--host", "0.0.0.0",
    "--port", "6969"
])