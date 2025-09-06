#!/usr/bin/env python3
"""
Comprehensive test runner for the crypto trading system.

This script provides a centralized way to run all tests with different options,
generate coverage reports, and validate the entire system.
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path


def run_command(command, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=capture_output,
            text=True,
            cwd=Path(__file__).parent
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def run_unit_tests(verbose=False, fail_fast=False):
    """Run all unit tests."""
    print_banner("RUNNING UNIT TESTS")
    
    test_files = [
        "tests/test_config.py",
        "tests/test_data_fetcher.py", 
        "tests/test_data_processor.py",
        "tests/test_strategies.py",
        "tests/test_backtesting_engine.py"
    ]
    
    cmd = ["python", "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    if fail_fast:
        cmd.append("-x")
    
    cmd.extend(test_files)
    
    print(f"Running command: {' '.join(cmd)}")
    return_code, stdout, stderr = run_command(' '.join(cmd))
    
    if return_code == 0:
        print("✅ All unit tests passed!")
    else:
        print("❌ Some unit tests failed!")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    
    return return_code == 0


def run_integration_tests(verbose=False, fail_fast=False):
    """Run integration tests."""
    print_banner("RUNNING INTEGRATION TESTS")
    
    cmd = ["python", "-m", "pytest", "tests/test_integration.py"]
    if verbose:
        cmd.append("-v")
    if fail_fast:
        cmd.append("-x")
    
    print(f"Running command: {' '.join(cmd)}")
    return_code, stdout, stderr = run_command(' '.join(cmd))
    
    if return_code == 0:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    
    return return_code == 0


def run_edge_case_tests(verbose=False, fail_fast=False):
    """Run edge case and error handling tests."""
    print_banner("RUNNING EDGE CASE TESTS")
    
    cmd = ["python", "-m", "pytest", "tests/test_edge_cases.py"]
    if verbose:
        cmd.append("-v")
    if fail_fast:
        cmd.append("-x")
    
    print(f"Running command: {' '.join(cmd)}")
    return_code, stdout, stderr = run_command(' '.join(cmd))
    
    if return_code == 0:
        print("✅ All edge case tests passed!")
    else:
        print("❌ Some edge case tests failed!")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    
    return return_code == 0


def run_performance_tests(verbose=False, fail_fast=False):
    """Run performance and load tests."""
    print_banner("RUNNING PERFORMANCE TESTS")
    
    cmd = ["python", "-m", "pytest", "tests/test_performance.py"]
    if verbose:
        cmd.append("-v")
    if fail_fast:
        cmd.append("-x")
    
    # Performance tests may take longer
    cmd.extend(["--tb=short", "-q"])
    
    print(f"Running command: {' '.join(cmd)}")
    print("⏳ Performance tests may take several minutes...")
    
    return_code, stdout, stderr = run_command(' '.join(cmd))
    
    if return_code == 0:
        print("✅ All performance tests passed!")
    else:
        print("❌ Some performance tests failed!")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    
    return return_code == 0


def run_coverage_report():
    """Generate and display test coverage report."""
    print_banner("GENERATING COVERAGE REPORT")
    
    # Install coverage if not present
    print("📦 Installing coverage package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], 
                  capture_output=True)
    
    # Run tests with coverage
    print("🔍 Running tests with coverage analysis...")
    cmd = [
        "python", "-m", "coverage", "run", "--source=src", 
        "-m", "pytest", "tests/", "-q"
    ]
    
    return_code, _, stderr = run_command(' '.join(cmd))
    
    if return_code != 0:
        print("❌ Failed to run coverage analysis!")
        print("STDERR:", stderr)
        return False
    
    # Generate coverage report
    print("📊 Generating coverage report...")
    return_code, stdout, stderr = run_command("python -m coverage report -m")
    
    if return_code == 0:
        print("📈 COVERAGE REPORT:")
        print(stdout)
        
        # Generate HTML report
        html_code, _, _ = run_command("python -m coverage html")
        if html_code == 0:
            print("📁 HTML coverage report generated in 'htmlcov/' directory")
    else:
        print("❌ Failed to generate coverage report!")
        print("STDERR:", stderr)
    
    return return_code == 0


def validate_test_environment():
    """Validate that the test environment is properly set up."""
    print_banner("VALIDATING TEST ENVIRONMENT")
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        "pytest", "pandas", "numpy", "requests", "yaml", "sqlite3"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "sqlite3":
                import sqlite3
            elif package == "yaml":
                import yaml  # pyyaml imports as 'yaml'
            else:
                __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            if package == "yaml":
                missing_packages.append("pyyaml")
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    # Check source code structure
    required_dirs = ["src", "tests"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory - Found")
        else:
            print(f"❌ {dir_name}/ directory - Missing")
            return False
    
    # Check test files exist
    test_files = [
        "tests/test_config.py",
        "tests/test_data_fetcher.py",
        "tests/test_data_processor.py",
        "tests/test_strategies.py",
        "tests/test_backtesting_engine.py",
        "tests/test_integration.py",
        "tests/test_edge_cases.py",
        "tests/test_performance.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✅ {test_file} - Found")
        else:
            print(f"❌ {test_file} - Missing")
            return False
    
    print("\n✅ Test environment validation completed successfully!")
    return True


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print_banner("RUNNING QUICK SMOKE TEST")
    
    # Run a subset of fast tests
    cmd = [
        "python", "-m", "pytest", 
        "tests/test_config.py::TestConfig::test_config_loading_success",
        "tests/test_data_processor.py::TestDataProcessor::test_add_basic_indicators_success",
        "tests/test_strategies.py::TestSwingTradingStrategy::test_calculate_signal_hold",
        "-v", "--tb=short"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    return_code, stdout, stderr = run_command(' '.join(cmd))
    
    if return_code == 0:
        print("✅ Smoke test passed! Basic functionality is working.")
    else:
        print("❌ Smoke test failed! There may be basic setup issues.")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    
    return return_code == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for crypto trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit --integration     # Run unit and integration tests
  python run_tests.py --smoke                  # Quick smoke test
  python run_tests.py --coverage               # Run all tests with coverage
  python run_tests.py --validate               # Validate test environment
  python run_tests.py --performance            # Run performance tests only
        """
    )
    
    parser.add_argument("--all", action="store_true", 
                       help="Run all test suites")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests")
    parser.add_argument("--edge-cases", action="store_true",
                       help="Run edge case tests")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance tests")
    parser.add_argument("--smoke", action="store_true",
                       help="Run quick smoke test")
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--validate", action="store_true",
                       help="Validate test environment")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--fail-fast", "-x", action="store_true",
                       help="Stop on first failure")
    
    args = parser.parse_args()
    
    # If no specific test type is selected, show help
    if not any([args.all, args.unit, args.integration, args.edge_cases, 
                args.performance, args.smoke, args.coverage, args.validate]):
        parser.print_help()
        return 1
    
    start_time = time.time()
    all_passed = True
    
    # Validate environment first if requested
    if args.validate or args.all:
        if not validate_test_environment():
            return 1
    
    # Run smoke test if requested
    if args.smoke:
        if not run_quick_smoke_test():
            return 1
        return 0
    
    # Run test suites
    if args.unit or args.all:
        if not run_unit_tests(args.verbose, args.fail_fast):
            all_passed = False
            if args.fail_fast:
                return 1
    
    if args.integration or args.all:
        if not run_integration_tests(args.verbose, args.fail_fast):
            all_passed = False
            if args.fail_fast:
                return 1
    
    if args.edge_cases or args.all:
        if not run_edge_case_tests(args.verbose, args.fail_fast):
            all_passed = False
            if args.fail_fast:
                return 1
    
    if args.performance or args.all:
        if not run_performance_tests(args.verbose, args.fail_fast):
            all_passed = False
            if args.fail_fast:
                return 1
    
    # Generate coverage report if requested
    if args.coverage or args.all:
        if not run_coverage_report():
            all_passed = False
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print_banner("TEST SUMMARY")
    print(f"⏱️  Total execution time: {duration:.2f} seconds")
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! The crypto trading system is working correctly.")
        return 0
    else:
        print("❌ SOME TESTS FAILED! Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())