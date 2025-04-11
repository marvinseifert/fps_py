import paramiko
from scp import SCPClient
import os

# Configuration
pi_host = "139.184.160.151"
pi_user = "pi"
pi_password = "your_pi_password"  # Replace or use getpass() for safety
remote_capture_script = "python /home/pi/projects/server_sync/scripts/capture.py 5 100 /home/pi/projects/server_sync/out/cap1"
remote_output_dir = "/home/pi/projects/server_sync/out/cap1"
local_output_dir = r"C:\Users\Marvin\pi_images"
# Create local output dir if not exists
os.makedirs(local_output_dir, exist_ok=True)
# Step 1: SSH into Pi and run capture script
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting to Raspberry Pi...")
ssh.connect(pi_host, username=pi_user, password=pi_password)
# Activate venv and run script
print("Running capture script on Pi...")
commands = [
    "cd ~/projects/server_sync",
    "source venv/bin/activate",
    remote_capture_script,
]
stdin, stdout, stderr = ssh.exec_command(" && ".join(commands))
print(stdout.read().decode())
print(stderr.read().decode())
# Step 2: Use SCP to copy folder to Windows
print("Transferring images to Windows...")
scp = SCPClient(ssh.get_transport())
scp.get(remote_output_dir, local_path=local_output_dir, recursive=True)
# Step 3: Cleanup
scp.close()
ssh.close()
print("Done!")
