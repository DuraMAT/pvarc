import subprocess
import time
import os
import datetime

def kill_ptpcamerad_loop(seconds=5):
    print(f"🔫 Killing ptpcamerad repeatedly for {seconds} seconds...")
    start = time.time()
    while time.time() - start < seconds:
        try:
            # Get PID(s)
            output = subprocess.check_output(["pgrep", "ptpcamerad"]).decode().strip()
            for pid in output.splitlines():
                subprocess.run(["sudo", "kill", "-9", pid], check=False)
                print(f"🧨 Killed ptpcamerad (PID {pid})")
        except subprocess.CalledProcessError:
            pass  # no process found
        time.sleep(0.3)
    print("✅ Done trying to disable ptpcamerad")

def capture_photo(download_dir='~/Pictures/arc_photos'):
    download_dir = os.path.expanduser(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"canon_5dsr_photo_{timestamp}.jpg"
    filepath = os.path.join(download_dir, filename)

    # kill_ptpcamerad_loop(seconds=6)

    try:
        print("📸 Capturing image...")
        subprocess.run([
            "gphoto2",
            "--capture-image-and-download",
            f"--filename={filepath}"
        ], check=True)
        print(f"✅ Image saved to {filepath}")
    except subprocess.CalledProcessError as e:
        print("❌ Failed to capture image:", e)

if __name__ == "__main__":
    capture_photo()
