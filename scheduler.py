# scheduler.py  — paste this into a new file in your project folder
import subprocess, hashlib, time, os

DATA_FILE  = r"data\raw\boston.csv"
HASH_FILE  = ".last_hash"
INTERVAL   = 180   # 3 minutes

def get_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def save_hash(h):
    with open(HASH_FILE, "w") as f:
        f.write(h)

def load_hash():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            return f.read().strip()
    return ""

run_count = 0
print("Scheduler started. Running every 3 minutes.")
print("Press Ctrl+C to stop.\n")

while True:
    run_count += 1
    print(f"\n--- Run #{run_count} ---")

    # Step 1: Inject drift (odd runs) or reset (even runs)
    if run_count % 2 == 1:
        print("Injecting drift...")
        subprocess.run(["python", "scripts/drift_injector.py"], check=True)
    else:
        print("Resetting to clean data...")
        subprocess.run(["python", "scripts/drift_injector.py", "--reset"], check=True)

    # Step 2: Check if file changed
    current_hash = get_hash(DATA_FILE)
    last_hash    = load_hash()

    if current_hash != last_hash:
        print("Data changed! Running pipeline...")
        save_hash(current_hash)
        subprocess.run(["dvc", "repro"], check=True)
        print("Pipeline complete.")
    else:
        print("No change detected. Skipping.")

    print(f"Sleeping {INTERVAL} seconds...")
    time.sleep(INTERVAL)