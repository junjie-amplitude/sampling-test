#!/usr/bin/env python3
"""
xxHash32 vs SplitMix32 vs CustomHash Sampling Benchmark

This script benchmarks xxHash32, SplitMix32, and a custom hash algorithm
(similar to the Java implementation) to compare their distribution uniformity
when used for sampling. It tests various sample rates against Unix timestamps
from specified date ranges using parallel processing.

Requirements:
    - xxhash: pip install xxhash
"""

import xxhash
import calendar
import datetime
import statistics
import time
import multiprocessing as mp
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Configuration - modify these values as needed
START_DATE = "2025-06-02"  # Start date in YYYY-MM-DD format
END_DATE = "2025-06-02"    # End date in YYYY-MM-DD format
SAMPLE_INTERVAL = 1     # Sample every 1ms

def splitmix32(x: int) -> int:
    """
    SplitMix32 hash function implementation.

    Args:
        x: Input value to hash

    Returns:
        32-bit hash value
    """
    x = x & 0xFFFFFFFF  # Ensure 32-bit
    x ^= x >> 16
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def generate_hash_code(s: str) -> int:
    """
    Custom hash function implementation (similar to djb2 variant).
    Java equivalent: generateHashCode() method.

    Args:
        s: Input string to hash

    Returns:
        Hash value as integer
    """
    hash_val = 0
    if not s:
        return hash_val

    for char in s:
        chr_val = ord(char)
        hash_val = (hash_val << 5) - hash_val + chr_val
        # Keep within reasonable bounds (simulate Java int behavior)
        hash_val = hash_val & 0xFFFFFFFFFFFFFFFF  # 64-bit limit

    return hash_val

def custom_hash_for_sampling(timestamp: int) -> int:
    """
    Custom hash function adapted for sampling (returns raw hash value).

    Args:
        timestamp: Input timestamp to hash

    Returns:
        Raw hash value (before modulo operation)
    """
    hash_number = generate_hash_code(str(timestamp))
    abs_hash = abs(hash_number)
    abs_hash_multiply = abs_hash * 31
    return abs_hash_multiply


def compute_hash_chunk(timestamps_chunk: List[int]) -> List[Tuple[int, int, int]]:
    """
    Compute xxHash32, SplitMix32, and CustomHash for a chunk of timestamps.

    Args:
        timestamps_chunk: List of timestamps to hash

    Returns:
        List of tuples (xxhash32_value, splitmix32_value, custom_hash_value)
    """
    results = []
    for timestamp in timestamps_chunk:
        # xxHash32
        xxhash32_value = xxhash.xxh32(str(timestamp).encode()).intdigest()

        # SplitMix32
        splitmix32_value = splitmix32(timestamp)

        # Custom Hash (from Java implementation)
        custom_hash_value = custom_hash_for_sampling(timestamp)

        results.append((xxhash32_value, splitmix32_value, custom_hash_value))

    return results


@dataclass
class SamplingResult:
    """Results from a sampling test."""
    sample_rate: float
    expected_count: int
    actual_count_xxhash32: int
    actual_count_splitmix32: int
    actual_count_custom: int
    difference_xxhash32: int
    difference_splitmix32: int
    difference_custom: int
    percentage_error_xxhash32: float
    percentage_error_splitmix32: float
    percentage_error_custom: float


@dataclass
class DateRangeResult:
    """Results for a date range."""
    start_date: str
    end_date: str
    timestamp_count: int
    sampling_results: List[SamplingResult]


class xxHashBenchmark:
    """Benchmark xxHash distribution for sampling applications."""

    def __init__(self, start_date: str, end_date: str, sample_interval: int = 1000):
        """
        Initialize the benchmark with a date range.

        Args:
            start_date: Starting date in YYYY-MM-DD format (e.g., "2024-01-01")
            end_date: Ending date in YYYY-MM-DD format (e.g., "2024-01-31")
            sample_interval: Sample every N milliseconds (default: 1000 = every second)
        """
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        self.sample_interval = sample_interval
        self.modulo = 1000000  # Hash modulo for distribution testing

    def get_date_range_timestamps(self, start_date: datetime.datetime, end_date: datetime.datetime) -> List[int]:
        """
        Generate all Unix timestamps in milliseconds for a given date range.

        Args:
            start_date: Starting date
            end_date: Ending date

        Returns:
            List of Unix timestamps in milliseconds for the date range
        """
        # Set end time to end of the day
        end_datetime = end_date.replace(hour=23, minute=59, second=59, microsecond=999000)

        # Convert to Unix timestamps in milliseconds
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_datetime.timestamp() * 1000)

        # Generate timestamps with the specified interval
        timestamps = list(range(start_ts, end_ts + 1, self.sample_interval))

        return timestamps

    def get_all_timestamps(self) -> List[int]:
        """Get all timestamps for the configured date range."""
        print(f"Generating timestamps for {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}...")
        return self.get_date_range_timestamps(self.start_date, self.end_date)

    def compute_hash_values(self, timestamps: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Compute hash values for all timestamps using parallel processing.

        Args:
            timestamps: List of Unix timestamps to hash

        Returns:
            Tuple of (xxhash32_values, splitmix32_values, custom_hash_values) all modulo self.modulo
        """
        if not timestamps:
            return [], [], []

        # Try parallel processing first, fall back to serial if it fails
        try:
            # Determine number of processes and chunk size
            num_processes = min(mp.cpu_count(), 4)  # Cap at 4 processes
            chunk_size = max(1000, len(timestamps) // num_processes)

            # Split timestamps into chunks
            chunks = [timestamps[i:i + chunk_size] for i in range(0, len(timestamps), chunk_size)]

            # Process in parallel
            with mp.Pool(processes=num_processes) as pool:
                chunk_results = pool.map(compute_hash_chunk, chunks)

            # Flatten results and apply modulo
            xxhash32_values = []
            splitmix32_values = []
            custom_hash_values = []

            for chunk_result in chunk_results:
                for xxhash32_val, splitmix32_val, custom_val in chunk_result:
                    xxhash32_values.append(xxhash32_val % self.modulo)
                    splitmix32_values.append(splitmix32_val % self.modulo)
                    custom_hash_values.append(custom_val % self.modulo)

            return xxhash32_values, splitmix32_values, custom_hash_values

        except Exception as e:
            print(f"  Warning: Parallel processing failed ({e}), falling back to serial processing...")
            # Fall back to serial processing
            return self._compute_hash_values_serial(timestamps)

    def _compute_hash_values_serial(self, timestamps: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Compute hash values serially as fallback.

        Args:
            timestamps: List of Unix timestamps to hash

        Returns:
            Tuple of (xxhash32_values, splitmix32_values, custom_hash_values) all modulo self.modulo
        """
        xxhash32_values = []
        splitmix32_values = []
        custom_hash_values = []

        for timestamp in timestamps:
            # xxHash32
            xxhash32_value = xxhash.xxh32(str(timestamp).encode()).intdigest()
            xxhash32_values.append(xxhash32_value % self.modulo)

            # SplitMix32
            splitmix32_value = splitmix32(timestamp)
            splitmix32_values.append(splitmix32_value % self.modulo)

            # Custom Hash
            custom_hash_value = custom_hash_for_sampling(timestamp)
            custom_hash_values.append(custom_hash_value % self.modulo)

        return xxhash32_values, splitmix32_values, custom_hash_values

    def test_all_sampling_rates(self, xxhash32_values: List[int], splitmix32_values: List[int], custom_hash_values: List[int]) -> List[SamplingResult]:
        """
        Test all sampling rates against pre-computed hash values.

        Args:
            xxhash32_values: List of xxHash32 values (modulo self.modulo)
            splitmix32_values: List of SplitMix32 values (modulo self.modulo)
            custom_hash_values: List of Custom hash values (modulo self.modulo)

        Returns:
            List of SamplingResult for each sample rate
        """
        sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []

        for sample_rate in sample_rates:
            threshold = int(sample_rate * self.modulo)

            # Count samples for xxHash32
            xxhash32_count = sum(1 for hash_val in xxhash32_values if hash_val < threshold)

            # Count samples for SplitMix32
            splitmix32_count = sum(1 for hash_val in splitmix32_values if hash_val < threshold)

            # Count samples for Custom Hash
            custom_count = sum(1 for hash_val in custom_hash_values if hash_val < threshold)

            expected_count = int(len(xxhash32_values) * sample_rate)

            # Calculate differences and errors for all algorithms
            difference_xxhash32 = xxhash32_count - expected_count
            difference_splitmix32 = splitmix32_count - expected_count
            difference_custom = custom_count - expected_count

            percentage_error_xxhash32 = (difference_xxhash32 / expected_count) * 100 if expected_count > 0 else 0
            percentage_error_splitmix32 = (difference_splitmix32 / expected_count) * 100 if expected_count > 0 else 0
            percentage_error_custom = (difference_custom / expected_count) * 100 if expected_count > 0 else 0

            results.append(SamplingResult(
                sample_rate=sample_rate,
                expected_count=expected_count,
                actual_count_xxhash32=xxhash32_count,
                actual_count_splitmix32=splitmix32_count,
                actual_count_custom=custom_count,
                difference_xxhash32=difference_xxhash32,
                difference_splitmix32=difference_splitmix32,
                difference_custom=difference_custom,
                percentage_error_xxhash32=percentage_error_xxhash32,
                percentage_error_splitmix32=percentage_error_splitmix32,
                percentage_error_custom=percentage_error_custom
            ))

        return results

    def run_benchmark(self) -> List[DateRangeResult]:
        """
        Run the complete benchmark for the specified date range.

        Returns:
            List of DateRangeResult objects (typically just one for the full range)
        """
        print("Starting xxHash32 vs SplitMix32 vs CustomHash Sampling Benchmark")
        print(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Modulo value: {self.modulo:,}")
        print(f"Parallel processing: enabled (up to {min(mp.cpu_count(), 8)} cores, with serial fallback)")
        print()

        date_results = []

        print(f"Processing {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}...")

        # Generate timestamps for the entire date range
        start_time = time.time()
        range_timestamps = self.get_date_range_timestamps(self.start_date, self.end_date)
        generation_time = time.time() - start_time

        print(f"  Generated {len(range_timestamps):,} timestamps in {generation_time:.2f} seconds")

        # Compute hash values for the entire range (all three algorithms in parallel)
        start_time = time.time()
        xxhash32_values, splitmix32_values, custom_hash_values = self.compute_hash_values(range_timestamps)
        hash_time = time.time() - start_time

        print(f"  Hash computation: {hash_time:.2f}s")

        # Test all sampling rates at once using pre-computed hash values
        start_time = time.time()
        range_sampling_results = self.test_all_sampling_rates(xxhash32_values, splitmix32_values, custom_hash_values)
        sampling_time = time.time() - start_time

        print(f"  Sampling analysis: {sampling_time:.2f}s")

        # Store date range result
        date_result = DateRangeResult(
            start_date=self.start_date.strftime('%Y-%m-%d'),
            end_date=self.end_date.strftime('%Y-%m-%d'),
            timestamp_count=len(range_timestamps),
            sampling_results=range_sampling_results
        )
        date_results.append(date_result)

        print(f"  Date range {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')} completed")
        print()

        return date_results

    def calculate_aggregated_results(self, date_results: List[DateRangeResult]) -> List[SamplingResult]:
        """
        Calculate aggregated sampling results across all date ranges.

        Args:
            date_results: List of individual date range results

        Returns:
            List of aggregated sampling results
        """
        if not date_results:
            return []

        # Aggregate sampling results
        sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        aggregated_sampling = []

        for i, rate in enumerate(sample_rates):
            total_expected = sum(result.sampling_results[i].expected_count for result in date_results)
            total_actual_xxhash32 = sum(result.sampling_results[i].actual_count_xxhash32 for result in date_results)
            total_actual_splitmix32 = sum(result.sampling_results[i].actual_count_splitmix32 for result in date_results)
            total_actual_custom = sum(result.sampling_results[i].actual_count_custom for result in date_results)

            total_difference_xxhash32 = total_actual_xxhash32 - total_expected
            total_difference_splitmix32 = total_actual_splitmix32 - total_expected
            total_difference_custom = total_actual_custom - total_expected

            percentage_error_xxhash32 = (total_difference_xxhash32 / total_expected) * 100 if total_expected > 0 else 0
            percentage_error_splitmix32 = (total_difference_splitmix32 / total_expected) * 100 if total_expected > 0 else 0
            percentage_error_custom = (total_difference_custom / total_expected) * 100 if total_expected > 0 else 0

            aggregated_sampling.append(SamplingResult(
                sample_rate=rate,
                expected_count=total_expected,
                actual_count_xxhash32=total_actual_xxhash32,
                actual_count_splitmix32=total_actual_splitmix32,
                actual_count_custom=total_actual_custom,
                difference_xxhash32=total_difference_xxhash32,
                difference_splitmix32=total_difference_splitmix32,
                difference_custom=total_difference_custom,
                percentage_error_xxhash32=percentage_error_xxhash32,
                percentage_error_splitmix32=percentage_error_splitmix32,
                percentage_error_custom=percentage_error_custom
            ))

        return aggregated_sampling

    def print_date_range_summary(self, date_results: List[DateRangeResult]):
        """Print a summary table of all date ranges."""
        print("=" * 160)
        print("DATE RANGE SUMMARY")
        print("=" * 160)
        print(f"{'Date Range':<25} {'Timestamps':<12} {'xxHash32 Avg %':<15} {'xxHash32 Max %':<15} {'SplitMix32 Avg %':<17} {'SplitMix32 Max %':<17} {'Custom Avg %':<14} {'Custom Max %':<14}")
        print("-" * 160)

        for result in date_results:
            xxhash32_errors = [abs(r.percentage_error_xxhash32) for r in result.sampling_results]
            splitmix32_errors = [abs(r.percentage_error_splitmix32) for r in result.sampling_results]
            custom_errors = [abs(r.percentage_error_custom) for r in result.sampling_results]

            xxhash32_avg_error = sum(xxhash32_errors) / len(xxhash32_errors)
            xxhash32_max_error = max(xxhash32_errors)
            splitmix32_avg_error = sum(splitmix32_errors) / len(splitmix32_errors)
            splitmix32_max_error = max(splitmix32_errors)
            custom_avg_error = sum(custom_errors) / len(custom_errors)
            custom_max_error = max(custom_errors)

            date_range_str = f"{result.start_date} to {result.end_date}"
            print(f"{date_range_str:<25} {result.timestamp_count:>11,} "
                  f"{xxhash32_avg_error:>14.3f}% {xxhash32_max_error:>14.3f}% "
                  f"{splitmix32_avg_error:>16.3f}% {splitmix32_max_error:>16.3f}% "
                  f"{custom_avg_error:>13.3f}% {custom_max_error:>13.3f}%")

        print()

    def print_detailed_date_range_results(self, date_results: List[DateRangeResult]):
        """Print detailed results for each date range."""
        for result in date_results:
            print("=" * 130)
            print(f"DETAILED RESULTS FOR {result.start_date} to {result.end_date}")
            print("=" * 130)
            print(f"Timestamps: {result.timestamp_count:,}")
            print()

            # Sampling results for this date range
            print("Sampling Rate Analysis:")
            print(f"{'Rate':<6} {'Expected':<10} {'xxHash32':<12} {'xxH32 Err%':<12} {'SplitMix32':<12} {'SM32 Err%':<12} {'CustomHash':<12} {'Custom Err%':<12}")
            print("-" * 130)

            for sampling_result in result.sampling_results:
                print(f"{sampling_result.sample_rate*100:>5.0f}% "
                      f"{sampling_result.expected_count:>9,} "
                      f"{sampling_result.actual_count_xxhash32:>11,} "
                      f"{sampling_result.percentage_error_xxhash32:>+11.3f}% "
                      f"{sampling_result.actual_count_splitmix32:>11,} "
                      f"{sampling_result.percentage_error_splitmix32:>+11.3f}% "
                      f"{sampling_result.actual_count_custom:>11,} "
                      f"{sampling_result.percentage_error_custom:>+11.3f}%")

            print()

    def print_aggregated_results(self, sampling_results: List[SamplingResult]):
        """Print aggregated results across all algorithms."""

        print("=" * 160)
        print("AGGREGATED SAMPLING RATE ANALYSIS")
        print("=" * 160)
        print(f"{'Rate':<6} {'Expected':<12} {'xxHash32':<12} {'xxH32 Err%':<12} {'SplitMix32':<12} {'SM32 Err%':<12} {'CustomHash':<12} {'Custom Err%':<12} {'Winner':<10}")
        print("-" * 160)

        for result in sampling_results:
            xxh32_abs_err = abs(result.percentage_error_xxhash32)
            sm32_abs_err = abs(result.percentage_error_splitmix32)
            custom_abs_err = abs(result.percentage_error_custom)

            # Determine winner (algorithm with smallest absolute error)
            min_error = min(xxh32_abs_err, sm32_abs_err, custom_abs_err)
            if min_error == xxh32_abs_err and xxh32_abs_err < sm32_abs_err and xxh32_abs_err < custom_abs_err:
                winner = "xxHash32"
            elif min_error == sm32_abs_err and sm32_abs_err < xxh32_abs_err and sm32_abs_err < custom_abs_err:
                winner = "SplitMix32"
            elif min_error == custom_abs_err and custom_abs_err < xxh32_abs_err and custom_abs_err < sm32_abs_err:
                winner = "CustomHash"
            else:
                winner = "Tie"

            print(f"{result.sample_rate*100:>5.0f}% "
                  f"{result.expected_count:>11,} "
                  f"{result.actual_count_xxhash32:>11,} "
                  f"{result.percentage_error_xxhash32:>+11.3f}% "
                  f"{result.actual_count_splitmix32:>11,} "
                  f"{result.percentage_error_splitmix32:>+11.3f}% "
                  f"{result.actual_count_custom:>11,} "
                  f"{result.percentage_error_custom:>+11.3f}% "
                  f"{winner:<10}")

        print()
        print("=" * 160)


def main():
    """Main function to run the benchmark with configurable parameters."""
    print("xxHash32 vs SplitMix32 vs CustomHash Sampling Benchmark")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Sample interval: every {SAMPLE_INTERVAL:,} milliseconds")
    print(f"  Testing sample rates: 10%, 20%, 30%, ..., 90%")
    print(f"  Hash modulo: 1,000,000")
    print(f"  Parallel processing: enabled")
    print()

    # Create and run benchmark
    benchmark = xxHashBenchmark(START_DATE, END_DATE, SAMPLE_INTERVAL)
    date_results = benchmark.run_benchmark()

    # Calculate aggregated results
    print("Calculating aggregated results...")
    aggregated_sampling = benchmark.calculate_aggregated_results(date_results)
    print()

    # Print date range summary
    benchmark.print_date_range_summary(date_results)

    # Print detailed results for each date range
    print("=" * 100)
    print("DETAILED DATE RANGE RESULTS")
    print("=" * 100)
    benchmark.print_detailed_date_range_results(date_results)

    # Print aggregated results
    benchmark.print_aggregated_results(aggregated_sampling)

    # Overall summary
    print("OVERALL SUMMARY:")
    print("=" * 50)

    # Calculate statistics across all date ranges
    all_xxhash32_errors = []
    all_splitmix32_errors = []
    all_custom_errors = []
    total_timestamps = 0

    for result in date_results:
        range_xxhash32_errors = [abs(r.percentage_error_xxhash32) for r in result.sampling_results]
        range_splitmix32_errors = [abs(r.percentage_error_splitmix32) for r in result.sampling_results]
        range_custom_errors = [abs(r.percentage_error_custom) for r in result.sampling_results]
        all_xxhash32_errors.extend(range_xxhash32_errors)
        all_splitmix32_errors.extend(range_splitmix32_errors)
        all_custom_errors.extend(range_custom_errors)
        total_timestamps += result.timestamp_count

    # Aggregated statistics
    overall_avg_error_xxhash32 = statistics.mean(all_xxhash32_errors)
    overall_max_error_xxhash32 = max(all_xxhash32_errors)
    overall_min_error_xxhash32 = min(all_xxhash32_errors)

    overall_avg_error_splitmix32 = statistics.mean(all_splitmix32_errors)
    overall_max_error_splitmix32 = max(all_splitmix32_errors)
    overall_min_error_splitmix32 = min(all_splitmix32_errors)

    overall_avg_error_custom = statistics.mean(all_custom_errors)
    overall_max_error_custom = max(all_custom_errors)
    overall_min_error_custom = min(all_custom_errors)

    aggregated_xxhash32_errors = [abs(r.percentage_error_xxhash32) for r in aggregated_sampling]
    aggregated_splitmix32_errors = [abs(r.percentage_error_splitmix32) for r in aggregated_sampling]
    aggregated_custom_errors = [abs(r.percentage_error_custom) for r in aggregated_sampling]

    aggregated_avg_error_xxhash32 = statistics.mean(aggregated_xxhash32_errors)
    aggregated_max_error_xxhash32 = max(aggregated_xxhash32_errors)

    aggregated_avg_error_splitmix32 = statistics.mean(aggregated_splitmix32_errors)
    aggregated_max_error_splitmix32 = max(aggregated_splitmix32_errors)

    aggregated_avg_error_custom = statistics.mean(aggregated_custom_errors)
    aggregated_max_error_custom = max(aggregated_custom_errors)

    print(f"Total date ranges tested: {len(date_results)}")
    print(f"Total timestamps: {total_timestamps:,}")
    print()

    print("Individual Date Range Statistics:")
    print(f"  xxHash32 - Avg: {overall_avg_error_xxhash32:.3f}%, Max: {overall_max_error_xxhash32:.3f}%, Min: {overall_min_error_xxhash32:.3f}%")
    print(f"  SplitMix32 - Avg: {overall_avg_error_splitmix32:.3f}%, Max: {overall_max_error_splitmix32:.3f}%, Min: {overall_min_error_splitmix32:.3f}%")
    print(f"  CustomHash - Avg: {overall_avg_error_custom:.3f}%, Max: {overall_max_error_custom:.3f}%, Min: {overall_min_error_custom:.3f}%")
    print()

    print("Aggregated Statistics:")
    print(f"  xxHash32 - Avg: {aggregated_avg_error_xxhash32:.3f}%, Max: {aggregated_max_error_xxhash32:.3f}%")
    print(f"  SplitMix32 - Avg: {aggregated_avg_error_splitmix32:.3f}%, Max: {aggregated_max_error_splitmix32:.3f}%")
    print(f"  CustomHash - Avg: {aggregated_avg_error_custom:.3f}%, Max: {aggregated_max_error_custom:.3f}%")
    print()

    # Winner analysis with three algorithms
    xxhash32_wins = 0
    splitmix32_wins = 0
    custom_wins = 0
    ties = 0

    for r in aggregated_sampling:
        xxh32_err = abs(r.percentage_error_xxhash32)
        sm32_err = abs(r.percentage_error_splitmix32)
        custom_err = abs(r.percentage_error_custom)

        min_error = min(xxh32_err, sm32_err, custom_err)

        if min_error == xxh32_err and xxh32_err < sm32_err and xxh32_err < custom_err:
            xxhash32_wins += 1
        elif min_error == sm32_err and sm32_err < xxh32_err and sm32_err < custom_err:
            splitmix32_wins += 1
        elif min_error == custom_err and custom_err < xxh32_err and custom_err < sm32_err:
            custom_wins += 1
        else:
            ties += 1

    print("Sampling Rate Winner Analysis:")
    print(f"  xxHash32 wins: {xxhash32_wins}/9 rates")
    print(f"  SplitMix32 wins: {splitmix32_wins}/9 rates")
    print(f"  CustomHash wins: {custom_wins}/9 rates")
    print(f"  Ties: {ties}/9 rates")
    print()

    # Quality assessment
    better_max_error = min(aggregated_max_error_xxhash32, aggregated_max_error_splitmix32, aggregated_max_error_custom)
    if better_max_error < 0.5:
        print("  ✓ Excellent distribution uniformity (best max aggregated error < 0.5%)")
    elif better_max_error < 1.0:
        print("  ✓ Very good distribution uniformity (best max aggregated error < 1%)")
    elif better_max_error < 2.0:
        print("  ✓ Good distribution uniformity (best max aggregated error < 2%)")
    else:
        print("  ⚠ Distribution may have some bias (best max aggregated error >= 2%)")

    print()
    print("Date range consistency:")
    range_avg_errors_xxhash32 = [statistics.mean([abs(r.percentage_error_xxhash32) for r in result.sampling_results])
                                for result in date_results]
    range_avg_errors_splitmix32 = [statistics.mean([abs(r.percentage_error_splitmix32) for r in result.sampling_results])
                                  for result in date_results]
    range_avg_errors_custom = [statistics.mean([abs(r.percentage_error_custom) for r in result.sampling_results])
                              for result in date_results]

    consistency_xxhash32 = statistics.stdev(range_avg_errors_xxhash32) if len(range_avg_errors_xxhash32) > 1 else 0
    consistency_splitmix32 = statistics.stdev(range_avg_errors_splitmix32) if len(range_avg_errors_splitmix32) > 1 else 0
    consistency_custom = statistics.stdev(range_avg_errors_custom) if len(range_avg_errors_custom) > 1 else 0

    print(f"  xxHash32 std dev: {consistency_xxhash32:.3f}%")
    print(f"  SplitMix32 std dev: {consistency_splitmix32:.3f}%")
    print(f"  CustomHash std dev: {consistency_custom:.3f}%")

    better_consistency = min(consistency_xxhash32, consistency_splitmix32, consistency_custom)
    if better_consistency < 0.1:
        print("  ✓ Excellent consistency across date ranges")
    elif better_consistency < 0.2:
        print("  ✓ Good consistency across date ranges")
    else:
        print("  ⚠ Some variation between date ranges")


if __name__ == "__main__":
    main()
