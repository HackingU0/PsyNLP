import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path


def print_step(message: str):
    print(f"\n{'=' * 60}")
    print(f"  {message}")
    print(f"{'=' * 60}\n")


def run_command(cmd: list[str], check: bool = True, shell: bool = False) -> bool:
    try:
        if shell:
            cmd_str = " ".join(cmd)
            result = subprocess.run(cmd_str, shell=True, check=check, capture_output=False)
        else:
            result = subprocess.run(cmd, check=check, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return False


def check_uv_installed() -> bool:
    return shutil.which("uv") is not None


def install_uv():
    print_step("Installing uv package manager")
    
    if check_uv_installed():
        print("uv is already installed")
        return True
    
    print("Installing uv...")
    system = platform.system()
    
    if system in ["Linux", "Darwin"]:  # macOS is Darwin
        # Use official installer
        cmd = ["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"]
        if run_command(cmd, shell=True):
            print("uv installed successfully")
            return True
    elif system == "Windows":
        cmd = ["powershell", "-c", "irm https://astral.sh/uv/install.ps1 | iex"]
        if run_command(cmd, shell=True):
            print("uv installed successfully")
            return True
    
    print("Failed to install uv. Please install manually from https://github.com/astral-sh/uv")
    return False


def sync_dependencies():
    print_step("Installing project dependencies")
    
    if not check_uv_installed():
        print("uv is not installed")
        return False
    
    print("Running uv sync...")
    if run_command(["uv", "sync"]):
        print("Dependencies installed successfully")
        return True
    else:
        print("Failed to install dependencies")
        return False


def get_system_memory_gb() -> float:
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            # Use sysctl to get total memory
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True
            )
            memory_bytes = int(result.stdout.strip())
            return memory_bytes / (1024 ** 3)
        
        elif system == "Linux":
            # Read from /proc/meminfo
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # MemTotal is in KB
                        memory_kb = int(line.split()[1])
                        return memory_kb / (1024 ** 2)
        
        elif system == "Windows":
            # Use wmic command
            result = subprocess.run(
                ["wmic", "computersystem", "get", "totalphysicalmemory"],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split('\n')
            memory_bytes = int(lines[1].strip())
            return memory_bytes / (1024 ** 3)
    
    except Exception as e:
        print(f"Warning: Could not detect system memory: {e}")
        print("Defaulting to 8GB assumption (will download larger model)")
        return 8.0  # Default to assuming sufficient memory
    
    # Fallback
    print("Warning: Unknown platform, defaulting to 8GB assumption")
    return 8.0


def download_file(url: str, dest_path: Path, description: str = "file"):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        file_size = dest_path.stat().st_size
        # Check if file is suspiciously small
        if file_size < 1024 * 100:
            print(f"{description} exists but is only {file_size} bytes, re-downloading...")
            dest_path.unlink()
        else:
            print(f"{description} already exists at {dest_path} ({file_size / (1024**2):.1f} MB)")
            return True
    
    print(f"Downloading {description}...")
    print(f"  From: {url}")
    print(f"  To: {dest_path}")
    
    try:
        try:
            from huggingface_hub import hf_hub_download
            
            parts = url.split('/')
            if 'huggingface.co' in url:
                repo_idx = parts.index('huggingface.co') + 1
                repo_id = f"{parts[repo_idx]}/{parts[repo_idx + 1]}"
                filename = parts[-1].split('?')[0]
                
                print(f"  Using Hugging Face Hub API...")
                print(f"  Repository: {repo_id}")
                print(f"  Filename: {filename}")
                
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(dest_path.parent),
                    local_dir_use_symlinks=False
                )
                
                # Rename if needed
                if Path(downloaded_path).name != dest_path.name:
                    Path(downloaded_path).rename(dest_path)
                
                file_size = dest_path.stat().st_size
                print(f"Downloaded {description} successfully ({file_size / (1024**2):.1f} MB)")
                return True
        except ImportError:
            print("  huggingface_hub not available, trying wget/curl...")
        
        # Fallback to wget or curl
        if shutil.which("wget"):
            cmd = ["wget", "-O", str(dest_path), url]
            if run_command(cmd):
                file_size = dest_path.stat().st_size
                print(f"Downloaded {description} successfully ({file_size / (1024**2):.1f} MB)")
                return True
        elif shutil.which("curl"):
            cmd = ["curl", "-L", "-o", str(dest_path), url]
            if run_command(cmd):
                file_size = dest_path.stat().st_size
                print(f"Downloaded {description} successfully ({file_size / (1024**2):.1f} MB)")
                return True
        else:
            print("Neither wget nor curl available, and huggingface_hub not installed")
            return False
            
    except Exception as e:
        print(f"\nFailed to download {description}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial download
        return False


def download_llm_models():
    print_step("Downloading LLM Models")
    
    memory_gb = get_system_memory_gb()
    print(f"System memory: {memory_gb:.2f} GB")
    
    models_dir = Path("nlp_models/GGUFS")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if memory_gb < 6:
        print("Low memory detected: downloading Gemma-3 1B model")
        url = "https://huggingface.co/lmstudio-community/gemma-3-1B-it-qat-GGUF/resolve/main/gemma-3-1B-it-QAT-Q4_0.gguf"
        dest = models_dir / "gemma-3-1B-it-QAT-Q4_0.gguf"
        return download_file(url, dest, "Gemma-3 1B GGUF model")
    else:
        print("Sufficient memory detected: downloading Qwen3-4B Thinking model")
        url = "https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF/resolve/main/Qwen3-4B-Thinking-2507-IQ4_XS.gguf"
        dest = models_dir / "Qwen3-4B-Thinking-2507-IQ4_XS.gguf"
        return download_file(url, dest, "Qwen3-4B Thinking GGUF model")


def download_nlp_models():
    print_step("Downloading NLP Models")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        if not run_command([sys.executable, "-m", "pip", "install", "huggingface_hub"]):
            print("Failed to install huggingface_hub")
            return False
        from huggingface_hub import snapshot_download
    
    models_dir = Path("nlp_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    bert_emotion_path = models_dir / "bert-emotion"
    if bert_emotion_path.exists():
        print(f"bert-emotion already exists at {bert_emotion_path}")
    else:
        print("Downloading bert-emotion model...")
        try:
            snapshot_download(
                repo_id="boltuix/bert-emotion",
                local_dir=str(bert_emotion_path),
                local_dir_use_symlinks=False
            )
            print("Downloaded bert-emotion successfully")
        except Exception as e:
            print(f"Failed to download bert-emotion: {e}")
            return False
    
    deberta_illness_path = models_dir / "deberta_illness"
    if deberta_illness_path.exists():
        print(f"deberta_illness already exists at {deberta_illness_path}")
    else:
        print("Downloading deberta_mental model (will be renamed to deberta_illness)...")
        try:
            snapshot_download(
                repo_id="elishaw/deberta_mental",
                local_dir=str(deberta_illness_path),
                local_dir_use_symlinks=False
            )
            print("Downloaded and renamed to deberta_illness successfully")
        except Exception as e:
            print(f"Failed to download deberta_mental: {e}")
            return False
    
    return True


def verify_installation():
    print_step("Verifying Installation")
    
    required_paths = [
        "nlp_models/bert-emotion",
        "nlp_models/deberta_illness",
        "nlp_models/GGUFS",
    ]
    
    all_present = True
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"{path_str}")
        else:
            print(f"Missing: {path_str}")
            all_present = False
    
    # Check for at least one GGUF model
    gguf_dir = Path("nlp_models/GGUFS")
    if gguf_dir.exists():
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if gguf_files:
            print(f"Found {len(gguf_files)} GGUF model(s)")
            for gguf_file in gguf_files:
                print(f"  - {gguf_file.name}")
        else:
            print("No GGUF models found")
            all_present = False
    
    return all_present


def main():
    print("""
PsyNLP Automated Deployment Script

This script will:
1. Install uv package manager 
2. Install project dependencies (uv sync)
3. Download LLM models (based on system memory)
4. Download NLP models (BERT, DeBERTa)
""")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {script_dir.absolute()}\n")
    
    steps = [
        ("Install uv", install_uv),
        ("Sync dependencies", sync_dependencies),
        ("Download LLM models", download_llm_models),
        ("Download NLP models", download_nlp_models),
        ("Verify installation", verify_installation),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        try:
            if not step_func():
                failed_steps.append(step_name)
                print(f"\nStep '{step_name}' completed with issues")
        except Exception as e:
            failed_steps.append(step_name)
            print(f"\nStep '{step_name}' failed with error: {e}")
    
    print("\n" + "=" * 60)
    if not failed_steps:
        print("Deployment completed successfully!")
        print("\nYou can now run the application with:")
        print("  uv run streamlit run app.py")
    else:
        print("Deployment completed with some issues:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease review the errors above and retry if needed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDeployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
