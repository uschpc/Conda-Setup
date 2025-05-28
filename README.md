# CARC Conda Environment Setup Script

This script (`carc_conda_setup.py`) is an interactive utility to help users set up a Conda environment on the USC Center for Advanced Research Computing (CARC) systems. It automates common tasks like environment creation, package installation, and configuration setup, providing a user-friendly, step-by-step experience.

---

## 📦 Features

- Interactive setup for Conda environments
- Automatic detection of first-time runs
- Environment configuration storage
- Streamlined Conda environment creation and activation
- Package installation from YAML or user-specified inputs
- Designed for USC CARC users

---

## 🖥️ Requirements

- A CARC account
- Python 3.x
- Conda installed and available on your shell
- Internet access (for fetching packages)

---

## ⚙️ Usage

1. **SSH into CARC:**

   ```bash
   ssh [your-username]@discovery.usc.edu
   ```

2. **Run the script:**

   ```bash
   python3 carc_conda_setup.py
   ```

   The script will walk you through the setup interactively.

3. **Re-run support:**

   The script tracks whether it’s your first time running it using a hidden config file stored at `~/.carc_conda_config.json`. If already initialized, it will skip some steps to avoid redundant setup.

---

## 🛠️ Script Workflow

1. Check if this is the first run using a config file.
2. Prompt the user for environment name and desired packages or YAML file.
3. Create and activate the Conda environment.
4. Install packages and verify.
5. Log setup success and store metadata.
6. Optionally configure shell startup files to auto-activate the environment.

---

## 📁 Example

```bash
$ python3 carc_conda_setup.py
[Step 1/6] Checking existing config...
[Step 2/6] Enter a name for your conda environment: myenv
[Step 3/6] Provide packages (e.g. numpy pandas) or YAML path: numpy pandas
...
Environment 'myenv' created and packages installed successfully!
```

---

## 🧼 Cleanup

To rerun the script from scratch:

```bash
rm ~/.carc_conda_config.json
```

---

## 📄 License

This script is intended for academic use within USC CARC. Please cite or credit appropriately when used in shared environments or publications.
