#!/usr/bin/env python3
"""
Main entry point for running different scenarios.

This script provides a unified interface to run any of the five experimental scenarios:
1. Baseline - Normal colored images
2. Grayscale - Convert images to grayscale
3. Augmented - Data augmentation on colored images
4. Image Sizes - Test multiple image resolutions
5. Quality Comparison - Low vs high quality images
"""

import argparse
import sys
from pathlib import Path

# Import scenario runners
from scenario1_baseline import run_scenario1
from scenario2_grayscale import run_scenario2
from scenario3_augmented import run_scenario3
from scenario4_image_sizes import run_scenario4
from scenario5_quality_comparison import run_scenario5

from utils import print_section


def print_banner():
    """Print project banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       Fruit Identification and Quality Assessment             ║
    ║              Deep Learning CNN Project                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_scenarios():
    """Print available scenarios."""
    print("\nAvailable Scenarios:")
    print("-" * 70)
    print("  1. Baseline - Normal colored images")
    print("     Establish baseline performance with standard RGB images")
    print()
    print("  2. Grayscale - Convert images to grayscale")
    print("     Test if color information is critical for classification")
    print()
    print("  3. Augmented - Data augmentation on colored images")
    print("     Apply augmentation to improve model generalization")
    print()
    print("  4. Image Sizes - Test multiple image resolutions")
    print("     Find optimal balance between size and performance")
    print("     Tests: 64x64, 128x128, 224x224, 299x299")
    print()
    print("  5. Quality Comparison - Low vs high quality images")
    print("     Compare model robustness across quality levels")
    print()
    print("  all - Run all scenarios sequentially")
    print("-" * 70)


def run_scenario(scenario_num):
    """
    Run a specific scenario.

    Args:
        scenario_num (int): Scenario number (1-5)

    Returns:
        dict: Results from the scenario
    """
    scenario_runners = {
        1: run_scenario1,
        2: run_scenario2,
        3: run_scenario3,
        4: run_scenario4,
        5: run_scenario5
    }

    if scenario_num not in scenario_runners:
        print(f"Error: Invalid scenario number {scenario_num}")
        return None

    print(f"\nRunning Scenario {scenario_num}...")
    return scenario_runners[scenario_num]()


def run_all_scenarios():
    """Run all scenarios sequentially."""
    print_section("RUNNING ALL SCENARIOS")

    all_results = {}

    for scenario_num in range(1, 6):
        print(f"\n{'='*70}")
        print(f"Starting Scenario {scenario_num}/5")
        print(f"{'='*70}\n")

        try:
            result = run_scenario(scenario_num)
            if result:
                all_results[scenario_num] = result
                print(f"\nScenario {scenario_num} completed successfully!")
            else:
                print(f"\nScenario {scenario_num} failed or was skipped.")
        except Exception as e:
            print(f"\nError in Scenario {scenario_num}: {str(e)}")
            continue

    # Print summary
    print_section("ALL SCENARIOS COMPLETED")
    print("\nSummary of Results:")
    print("-" * 70)

    for scenario_num, result in all_results.items():
        if 'val_metrics' in result:
            print(f"\nScenario {scenario_num}: {result['scenario_description']}")
            print(f"  Validation Accuracy: {result['val_metrics']['accuracy']:.4f}")
            print(f"  Validation F1-Score: {result['val_metrics']['f1_score']:.4f}")

    return all_results


def main():
    """Main function."""
    print_banner()

    parser = argparse.ArgumentParser(
        description="Run fruit quality assessment experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--scenario', '-s',
        type=str,
        choices=['1', '2', '3', '4', '5', 'all'],
        help='Scenario to run (1-5 or all)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available scenarios'
    )

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        print_scenarios()
        print("\nUsage:")
        print("  python main.py --scenario 1           # Run scenario 1")
        print("  python main.py --scenario all         # Run all scenarios")
        print("  python main.py --list                 # List all scenarios")
        print("\nFor more options:")
        print("  python main.py --help")
        return

    # List scenarios
    if args.list:
        print_scenarios()
        return

    # Run scenario
    if args.scenario:
        if args.scenario == 'all':
            results = run_all_scenarios()
        else:
            scenario_num = int(args.scenario)
            results = run_scenario(scenario_num)

            if results:
                print("\n" + "="*70)
                print("SCENARIO COMPLETED SUCCESSFULLY")
                print("="*70)

                if 'val_metrics' in results:
                    print(f"\nValidation Results:")
                    print(f"  Accuracy:  {results['val_metrics']['accuracy']:.4f}")
                    print(f"  Precision: {results['val_metrics']['precision']:.4f}")
                    print(f"  Recall:    {results['val_metrics']['recall']:.4f}")
                    print(f"  F1-Score:  {results['val_metrics']['f1_score']:.4f}")

                    if 'auc' in results['val_metrics'] and results['val_metrics']['auc']:
                        print(f"  AUC:       {results['val_metrics']['auc']:.4f}")

                print(f"\nResults saved in:")
                print(f"  - models/{results['scenario_name']}/")
                print(f"  - reports/{results['scenario_name']}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
