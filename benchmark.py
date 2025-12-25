"""
Performance Benchmarking Script for PsyNLP
Measures execution time and hardware specifications
"""
import timeit
import psutil
import platform
import torch
from datetime import datetime
import json
from modules.predict_score import calculate_text_severity, calculate_article_severity

def get_hardware_info():
    """Get system hardware information"""
    return {
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "device": "MPS" if torch.backends.mps.is_available() else "CUDA" if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__,
        "platform": platform.system(),
        "platform_version": platform.version(),
    }

def benchmark_text_severity(text_sample: str, number: int = 1):
    """
    Benchmark calculate_text_severity function
    Args:
        text_sample: Text to analyze
        number: How many times to run
    Returns:
        Timing results
    """
    # Prepare the function call
    def run_test():
        return calculate_text_severity(text_sample)
    
    # Run timeit
    print(f"Running benchmark {number} time(s)...")
    total_time = timeit.timeit(run_test, number=number)
    avg_time = total_time / number
    
    return {
        "total_time_seconds": round(total_time, 4),
        "average_time_seconds": round(avg_time, 4),
        "number_of_runs": number,
    }

def benchmark_article_severity(md_path: str, number: int = 1):
    """
    Benchmark calculate_article_severity function
    Args:
        md_path: Path to markdown file
        number: How many times to run
    Returns:
        Timing results
    """
    def run_test():
        return calculate_article_severity(md_path)
    
    print(f"Running benchmark {number} time(s)...")
    total_time = timeit.timeit(run_test, number=number)
    avg_time = total_time / number
    
    return {
        "total_time_seconds": round(total_time, 4),
        "average_time_seconds": round(avg_time, 4),
        "number_of_runs": number,
    }

def run_full_benchmark():
    """Run comprehensive benchmark"""
    print("=" * 60)
    print("PsyNLP Performance Benchmark")
    print("=" * 60)
    
    # Hardware info
    hardware = get_hardware_info()
    print("\nHardware Information:")
    print(f"  CPU: {hardware['cpu']}")
    print(f"  Cores: {hardware['cpu_count']} (Physical), {hardware['cpu_count_logical']} (Logical)")
    print(f"  RAM: {hardware['ram_gb']} GB")
    print(f"  Device: {hardware['device']}")
    print(f"  Platform: {hardware['platform']} {hardware['platform_version']}")
    print(f"  PyTorch: {hardware['pytorch_version']}")
    
    # Test sample text
    test_text = """
    I've been feeling really down lately. Nothing seems to interest me anymore.
    I wake up in the morning and just feel empty. Sometimes I wonder if it's worth going on.
    My friends try to help but I feel hopeless about everything.
    """
    
    # Benchmark text severity
    print("\n" + "=" * 60)
    print("Benchmarking calculate_text_severity()...")
    print("=" * 60)
    text_results = benchmark_text_severity(test_text.strip(), number=3)
    print(f"  Total Time: {text_results['total_time_seconds']}s")
    print(f"  Average Time: {text_results['average_time_seconds']}s")
    print(f"  Runs: {text_results['number_of_runs']}")
    
    # Benchmark article severity (if sample_text.md exists)
    try:
        print("\n" + "=" * 60)
        print("Benchmarking calculate_article_severity()...")
        print("=" * 60)
        article_results = benchmark_article_severity("sample_text.md", number=2)
        print(f"  Total Time: {article_results['total_time_seconds']}s")
        print(f"  Average Time: {article_results['average_time_seconds']}s")
        print(f"  Runs: {article_results['number_of_runs']}")
    except FileNotFoundError:
        print("  sample_text.md not found, skipping article benchmark")
        article_results = None
    
    # Summary report
    report = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hardware,
        "benchmarks": {
            "text_severity": text_results,
            "article_severity": article_results,
        }
    }
    
    # Save to file
    with open("benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Benchmark complete! Results saved to benchmark_results.json")
    print("=" * 60)

if __name__ == "__main__":
    run_full_benchmark()