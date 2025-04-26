# **bwf-vid-bench: AI Video Generation Benchmark Framework**

A minimal viable testing framework designed to benchmark AI video generation models, optimized for GPU acceleration on Linux systems.

The framework prioritizes reproducibility, automation via Taskfile, and resource efficiency, tailored to the target hardware.

## **Prerequisites**

* Operating System: Linux (or compatible)
* Python: 3.12+
* Git
* [Task](https://taskfile.dev/installation/) (Taskfile runner)
* NVIDIA GPU Drivers (compatible with GeForce/Quadro/Tesla NVIDIA cards and PyTorch/CUDA)
* Required system build dependencies for Python packages (e.g., `python3-devel`, `gcc`, etc.
  - check specific package requirements if installation fails).

## **Setup**

**Clone the repository and Setup Environment:**
   ```bash
   git clone https://github.com/lewars/bwf-vid-benchmark.git
   cd bwf-vid-bench
   pip install task
   task setup
   ```

## **Usage**

1. **Define Test Cases:**

   Edit the `test_cases/test_cases.yaml` file to define the specific benchmark scenarios you want to run (model, prompt, resolution, seed, etc.).
2. **Run the Benchmark:**

   Execute the main benchmark script. This will iterate through the test cases defined in `test_cases.yaml`, generate videos, and record metrics in timestamped directories within `results/`.

   ```bash
   task run
   ```

   or use the alias:

    ```bash
    task benchmark
    ```

   *(This is also the default task, so running `task` alone will execute the benchmark).*

3. **Analyze Results:**

   Run the analysis script to process the latest results (or specify a directory) and generate plots/summaries.

    ```bash
    task analyze
    ```

## **Development Tasks**

* **Run Unit Tests:**

  Execute the test suite using pytest, including coverage reporting.

    ```bash
    task test
    ```

* **Run Linters:**

  Check Python code style (flake8) and YAML syntax (yamllint).

  ```bash
  task lint         # Run all linters
  task lint-py      # Run only Python linter
  task lint-yaml    # Run only YAML linter
  ```

* **Clean Up:**

  Remove the virtual environment (`.venv`) and temporary cache files. Does not remove results.

    ```bash
    task clean
    ```

## **Directory Structure**

* `src/`: Main Python source code for the framework.
* `tests/`: Unit tests (pytest).
* `test_cases/`: Configuration file(s) defining benchmark runs.
* `results/`: Output directory for generated videos and metrics (timestamped).
* `Taskfile.yml`: Defines automation tasks.
* `requirements.txt`: Python package dependencies.
