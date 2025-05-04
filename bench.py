import logging
from pathlib import Path
import sys

from orchestrator import BenchmarkOrchestrator

BASE_RESULTS_DIR = Path("results")
TEST_CASES_YAML = Path("test_cases/test_cases.yaml")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main():
    log.info("BWF benchmark script started.")

    # -- Define Argparse ---

    # --- Option 1: Standard Full Run ---
    try:
        log.info("Starting standard full benchmark run...")
        orchestrator = BenchmarkOrchestrator(
            test_cases_yaml_path=TEST_CASES_YAML,
            base_results_dir=BASE_RESULTS_DIR
        )
        orchestrator.run() # Use the convenience method
        log.info("Standard run finished.")
        sys.exit(0)

    except Exception as e:
        log.exception(f"An uncaught error occurred during the standard run: {e}")
        sys.exit(1)

    # Arguments from commandline should determine which logic to use
    # --- Option 2: Granular Control Example (e.g., run only specific tests) ---
    # Comment out Option 1 and uncomment this section to use granular control
    # orchestrator_granular = None
    # try:
    #     log.info("Starting granular benchmark run...")
    #     orchestrator_granular = BenchmarkOrchestrator(
    #         test_cases_yaml_path=TEST_CASES_YAML,
    #         base_results_dir=BASE_RESULTS_DIR
    #     )
    #     orchestrator_granular.setup() # Step 1: Setup
    #     # Example: Run only tests with specific IDs
    #     ids_to_run = ["test001", "test003"] # Get these from args or config
    #     orchestrator_granular.execute_test_cases(test_ids_to_run=ids_to_run) # Step 2: Execute specific
    #     orchestrator_granular.compile_summary() # Step 3: Compile summary for executed tests
    #     log.info("Granular run finished.")
    #     sys.exit(0)
    # except Exception as e:
    #     log.exception(f"An uncaught error occurred during the granular run: {e}")
    #     sys.exit(1)
    # finally:
    #      # Ensure cleanup happens even in granular mode if orchestrator was created
    #      if orchestrator_granular:
    #          orchestrator_granular.cleanup() # Step 4: Cleanup

if __name__ == "__main__":
    main()
