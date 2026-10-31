import subprocess
import datetime
import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def main():
    print("=" * 60)
    print("Syncing colab-sd to GitHub (wenjunii/colab-sd)...")
    print("=" * 60)
    
    # 1. Update toe and tox
    print("\n1. Rebuilding colab-sd.toe and StreamDiffusion_Remote.tox...")
    code, out, err = run_cmd("python build_clean_toe.py")
    if code != 0:
        print("Warning during rebuild:", err)
        
    # 2. Update notebook
    print("2. Regenerating StreamDiffusion_Colab_Server.ipynb...")
    code, out, err = run_cmd("python generate_notebook.py")
    if code != 0:
        print("Warning during notebook generation:", err)

    # 3. Check git changes
    code, status_out, _ = run_cmd("git status --porcelain")
    if not status_out:
        print("\n[OK] Everything is already up to date! Nothing to push.")
        print("Repo: https://github.com/wenjunii/colab-sd")
        return
        
    print("\n3. Staging and committing changes...")
    run_cmd("git add .")
    
    msg = sys.argv[1] if len(sys.argv) > 1 else f"Update {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    code, out, err = run_cmd(f'git commit -m "{msg}"')
    if code == 0:
        print(f"   Committed: {msg}")
    else:
        print("Commit output:", out or err)
        
    print("4. Pushing to GitHub (origin main)...")
    code, out, err = run_cmd("git push origin main")
    if code == 0:
        print("\n[SUCCESS] Successfully synced to GitHub!")
        print("URL: https://github.com/wenjunii/colab-sd")
    else:
        print("Push error:", err)

if __name__ == '__main__':
    main()
