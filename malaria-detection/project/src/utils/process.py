import subprocess
import signal
import os


def check_port_in_use(port):
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        return len(result.stdout.strip().split("\n")) > 1
    except Exception as e:
        print(f"Error checking port {port}: {e}")
        return False


def kill_process_on_port(port):
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split("\n")
        for line in lines[1:]:
            parts = line.split()
            pid = int(parts[1])
            os.kill(pid, signal.SIGKILL)
            print(f"Killed process {pid} on port {port}")
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
