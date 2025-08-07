# Installing Electrolyzer

## Installing from Source

For most use cases, installing from source will be the preferred installation route.

1. Using Git, navigate to a local target directory and clone repository:

    ```bash
    git clone https://github.com/NREL/electrolyzer.git
    ```

2. Navigate to `electrolyzer`

    ```bash
    cd electrolyzer
    ```

3. Create a new virtual environment and change to it. Using Conda Python 3.11 (choose your favorite
   supported version) and naming it 'electrolyzer_env' (choose your desired name):

    ```bash
    conda create --name electrolyzer_env python=3.11 -y
    conda activate electrolyzer_env
    ```

4. Install electrolyzer and its dependencies:

    - If you want to just use electrolyzer:

       ```bash
       pip install .
       ```

    - If you want to work with the examples:

       ```bash
       pip install ".[examples]"
       ```

    - If you also want development dependencies for running tests and building docs:

       ```bash
       pip install -e ".[develop]"
       ```

    - In one step, all dependencies can be installed as:

      ```bash
      pip install -e ".[all]"
      ```

## Developer Notes

Developers should add install using `pip install -e ".[all]"` to ensure documentation testing, and
linting can be done without any additional installation steps.

Please be sure to also install the pre-commit hooks if contributing code back to the main
repository via the following. This enables a series of automated formatting and code linting
(style and correctness checking) to ensure the code is stylistically consistent.

```bash
pre-commit install
```

If a check (or multiple) fails (commit is blocked), and reformatting was done, then restage
(`git add`) your files and commit them again to see if all issues were resolved without user
intervention. If changes are required follow the suggested fix, or resolve the stated
issue(s). Restaging and committing may take multiple attempts steps if errors are unaddressed
or insufficiently addressed. Please see [pre-commit](https://pre-commit.com/),
[ruff](https://docs.astral.sh/ruff/), or [isort](https://pycqa.github.io/isort/) for more
information.
