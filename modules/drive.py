import os
import sys

def mount_drive():
    """Mounts Google Drive if running in Colab."""
    if "google.colab" in sys.modules:
        from google.colab import drive
        if not os.path.isdir('/content/drive/My Drive'):
            drive.mount('/content/drive')
            print("✅ Google Drive mounted successfully.")
        else:
            print("📂 Google Drive is already mounted.")
    else:
        print("⚠️ Not running in Google Colab. Skipping Google Drive mount.")

def setup_dataset(local_path="path/to/your/local/490data.tar.gz"):
    """
    Ensures dataset is available and extracted.

    Args:
        local_path (str): Path to dataset when running outside Colab.
    """
    IN_COLAB = "google.colab" in sys.modules
    DRIVE_PATH = "/content/drive/My Drive/490data.tar.gz" if IN_COLAB else local_path
    DEST_PATH = "/content/490data.tar.gz" if IN_COLAB else "490data.tar.gz"

    if IN_COLAB and not os.path.exists("/content"):
        os.makedirs("/content", exist_ok=True)

    if not os.path.exists(DEST_PATH):
        print(f"📥 Copying dataset from {DRIVE_PATH}...")
        os.system(f'cp "{DRIVE_PATH}" "{DEST_PATH}"')

    if os.path.exists("data"):
        print("🗑️ Removing old data folder...")
        os.system("rm -rf data")

    os.makedirs("data", exist_ok=True)

    print("📂 Extracting dataset...")
    os.system(f"tar -xvzf {DEST_PATH} -C .")

    os.remove(DEST_PATH)
    print("✅ Dataset setup completed!")
