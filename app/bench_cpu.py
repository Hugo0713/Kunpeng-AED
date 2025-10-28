"""
CPU benchmark tool for measuring inference performance
with different thread counts.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import psutil

from app.infer_tflite import YAMNetInference

logger = logging.getLogger(__name__)


class CPUBenchmark:
    """Benchmark tool for CPU inference performance."""
    
    def __init__(
        self,
        model_path: Path,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ):
        """
        Initialize benchmark.
        
        Args:
            model_path: Path to TFLite model
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
        """
        self.model_path = model_path
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        
        # Generate dummy input (64 Mel bands × 100 frames)
        self.dummy_input = np.random.randn(64, 100).astype(np.float32)
    
    def run(self, thread_counts: List[int]) -> dict:
        """
        Run benchmark for different thread counts.
        
        Args:
            thread_counts: List of thread counts to test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for num_threads in thread_counts:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking with {num_threads} thread(s)")
            logger.info(f"{'='*60}")
            
            # Initialize engine
            engine = YAMNetInference(self.model_path, num_threads=num_threads)
            
            # Warmup
            logger.info(f"Warmup: {self.warmup_iterations} iterations...")
            for _ in range(self.warmup_iterations):
                engine.predict(self.dummy_input)
            
            # Benchmark
            logger.info(f"Running benchmark: {self.num_iterations} iterations...")
            latencies = []
            cpu_usage = []
            
            process = psutil.Process()
            start_time = time.time()
            
            for i in range(self.num_iterations):
                # Measure CPU before inference
                cpu_percent = process.cpu_percent(interval=None)
                
                # Run inference
                _, latency_ms = engine.predict(self.dummy_input)
                latencies.append(latency_ms)
                cpu_usage.append(cpu_percent)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  Progress: {i+1}/{self.num_iterations}")
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            latencies = np.array(latencies)
            cpu_usage = np.array(cpu_usage)
            
            results[num_threads] = {
                'mean_latency': float(np.mean(latencies)),
                'p95_latency': float(np.percentile(latencies, 95)),
                'p99_latency': float(np.percentile(latencies, 99)),
                'min_latency': float(np.min(latencies)),
                'max_latency': float(np.max(latencies)),
                'qps': self.num_iterations / total_time,
                'mean_cpu': float(np.mean(cpu_usage)),
                'max_cpu': float(np.max(cpu_usage))
            }
            
            self._print_results(num_threads, results[num_threads])
        
        self._print_summary(results)
        return results
    
    def _print_results(self, num_threads: int, stats: dict) -> None:
        """Print benchmark results for a single thread count."""
        logger.info(f"\nResults for {num_threads} thread(s):")
        logger.info(f"  Mean latency:  {stats['mean_latency']:7.2f} ms")
        logger.info(f"  P95 latency:   {stats['p95_latency']:7.2f} ms")
        logger.info(f"  P99 latency:   {stats['p99_latency']:7.2f} ms")
        logger.info(f"  Min latency:   {stats['min_latency']:7.2f} ms")
        logger.info(f"  Max latency:   {stats['max_latency']:7.2f} ms")
        logger.info(f"  QPS:           {stats['qps']:7.2f}")
        logger.info(f"  Mean CPU:      {stats['mean_cpu']:7.2f} %")
        logger.info(f"  Max CPU:       {stats['max_cpu']:7.2f} %")
    
    def _print_summary(self, results: dict) -> None:
        """Print summary table comparing all thread counts."""
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'='*80}")
        
        header = f"{'Threads':>8} | {'Mean (ms)':>10} | {'P95 (ms)':>10} | {'QPS':>8} | {'CPU %':>8}"
        logger.info(header)
        logger.info("-" * 80)
        
        for num_threads, stats in sorted(results.items()):
            row = (
                f"{num_threads:>8} | "
                f"{stats['mean_latency']:>10.2f} | "
                f"{stats['p95_latency']:>10.2f} | "
                f"{stats['qps']:>8.2f} | "
                f"{stats['mean_cpu']:>8.2f}"
            )
            logger.info(row)
        
        logger.info("=" * 80)


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="CPU benchmark for YAMNet inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/x:/VS code/2025 Autumn/kunpeng/models/yamnet_int8.tflite"),
        help="Path to TFLite model"
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Thread counts to benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        return
    
    benchmark = CPUBenchmark(
        model_path=args.model,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    results = benchmark.run(args.threads)
    
    # Optionally save results to JSON
    import json
    output_path = Path("benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
