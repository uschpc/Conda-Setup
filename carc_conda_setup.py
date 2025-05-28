#!/usr/bin/env python3
import subprocess
import sys
import os
import platform
import shutil
import argparse
from pathlib import Path
import time
import getpass
import json

class InteractiveSetup:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 6
        self.status = {}
        self.config_file = os.path.expanduser("~/.carc_conda_config.json")
        self.is_first_time = self.check_first_time()

    def check_first_time(self):
        """Check if this is the first time running the script."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                return not config.get('initialized', False)
            except:
                return True
        return True

    def save_config(self):
        """Save configuration to mark initialization as complete."""
        config = {
            'initialized': True,
            'last_run': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user': getpass.getuser()
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def print_status(self, message, status="info"):
        """Print status message with color indicators."""
        if status == "success":
            print(f"✓ {message}")
        elif status == "error":
            print(f"✗ {message}")
        elif status == "warning":
            print(f"! {message}")
        elif status == "info":
            print(f"  {message}")
        elif status == "highlight":
            print(f"→ {message}")

    def get_user_input(self, prompt, default=None, required=False):
        """Get user input with a prompt."""
        while True:
            print(f"{prompt} ", end='')
            if default:
                print(f"[{default}] ", end='')
            user_input = input().strip()
            
            if not user_input and default:
                return default
            elif not user_input and required:
                self.print_status("This field is required. Please try again.", "warning")
                continue
            return user_input

    def show_first_time_welcome(self):
        """Show welcome message for first-time users."""
        print("Welcome to CARC Conda Environment Setup!")
        print("\nThis appears to be your first time using conda on CARC.")
        print("Let's get you set up with the basics:\n")
        
        print("1. We'll help you initialize conda in your environment")
        print("2. Set up your first conda environment")
        print("3. Install PyTorch and common data science packages")
        print("4. Create a sample job script for running your code\n")
        
        print("Press Enter to continue...")
        input()

    def show_welcome(self):
        """Show welcome message for returning users."""
        print("Welcome back to CARC Conda Environment Setup!")
        print("\nLet's create a new conda environment for your project.\n")
        
        print("Press Enter to continue...")
        input()

    def check_modules(self):
        """Check if module command is available and load conda module."""
        print("Loading conda module...")
        try:
            # Source the module command first
            module_cmd = "source /etc/profile.d/modules.sh && module purge && module load conda"
            subprocess.run(module_cmd, shell=True, check=True, executable='/bin/bash')
            print("Successfully loaded conda module")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error loading modules: {e}")
            return False

    def initialize_mamba(self):
        """Initialize Mamba in the shell."""
        print("Initializing Mamba...")
        try:
            if self.is_first_time:
                print("First time setup: Initializing Mamba in your shell...")
                subprocess.run(["mamba", "init", "bash"], check=True)
                print("Please run 'source ~/.bashrc' after this script completes")
            else:
                subprocess.run(["mamba", "init", "bash"], check=True)
                subprocess.run(["source", "~/.bashrc"], shell=True, check=True)
            print("Successfully initialized Mamba")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error initializing Mamba: {e}")
            return False

    def create_conda_env(self, env_name, python_version, project_dir=None):
        """Create a new conda environment."""
        print(f"Creating conda environment '{env_name}'...")
        try:
            if project_dir:
                env_path = Path(project_dir) / env_name
                print(f"Creating environment in project directory: {env_path}")
                subprocess.run(
                    ["mamba", "create", "--prefix", str(env_path), f"python={python_version}", "-y"],
                    check=True
                )
            else:
                subprocess.run(
                    ["mamba", "create", "--name", env_name, f"python={python_version}", "-y"],
                    check=True
                )
            print(f"Successfully created environment '{env_name}'")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating conda environment: {e}")
            return False

    def install_packages(self, env_name, project_dir=None):
        """Install required packages."""
        print("Installing packages...")
        
        # Initialize mamba first
        try:
            subprocess.run("mamba init bash", shell=True, check=True, executable='/bin/bash')
            # Source the bashrc to make mamba commands available
            subprocess.run("source ~/.bashrc", shell=True, check=True, executable='/bin/bash')
        except subprocess.CalledProcessError as e:
            print(f"Error initializing mamba: {e}")
            return False

        # Check for CUDA
        cuda_available = False
        try:
            nvidia_smi = shutil.which('nvidia-smi')
            if nvidia_smi:
                result = subprocess.run([nvidia_smi], capture_output=True, text=True)
                cuda_available = result.returncode == 0
        except:
            pass

        try:
            # Create a new environment first
            if project_dir:
                env_path = os.path.expanduser(f"{project_dir}/{env_name}")
                create_cmd = f"conda create -y -p {env_path} python={self.python_version}"
            else:
                create_cmd = f"conda create -y -n {env_name} python={self.python_version}"
            
            print(f"Creating new environment: {env_name}")
            subprocess.run(create_cmd, shell=True, check=True, executable='/bin/bash')

            # Install MKL first
            print("Installing MKL dependencies...")
            if project_dir:
                mkl_cmd = f"conda install -y -p {env_path} mkl mkl-include intel-openmp"
            else:
                mkl_cmd = f"conda install -y -n {env_name} mkl mkl-include intel-openmp"
            subprocess.run(mkl_cmd, shell=True, check=True, executable='/bin/bash')

            # Install PyTorch (required)
            if cuda_available:
                print("CUDA detected, installing PyTorch with CUDA support...")
                if project_dir:
                    install_cmd = f"conda install -y -p {env_path} pytorch pytorch-cuda=11.8 -c pytorch -c nvidia"
                else:
                    install_cmd = f"conda install -y -n {env_name} pytorch pytorch-cuda=11.8 -c pytorch -c nvidia"
            else:
                print("CUDA not detected, installing CPU-only PyTorch...")
                if project_dir:
                    install_cmd = f"conda install -y -p {env_path} pytorch cpuonly -c pytorch"
                else:
                    install_cmd = f"conda install -y -n {env_name} pytorch cpuonly -c pytorch"

            # Use conda for installation
            subprocess.run(install_cmd, shell=True, check=True, executable='/bin/bash')

            # Ask about additional packages
            print("\nWould you like to install additional packages?")
            print("Available packages:")
            print("1. torchvision (for computer vision)")
            print("2. torchaudio (for audio processing)")
            print("3. numpy (for numerical computing)")
            print("4. pandas (for data analysis)")
            print("5. scipy (for scientific computing)")
            print("6. scikit-learn (for machine learning)")
            print("7. matplotlib (for plotting)")
            print("8. jupyter (for notebooks)")
            
            packages_to_install = []
            while True:
                choice = self.get_user_input("\nEnter package number to install (or press Enter to finish):", "")
                if not choice:
                    break
                try:
                    choice = int(choice)
                    if 1 <= choice <= 8:
                        package_map = {
                            1: "torchvision",
                            2: "torchaudio",
                            3: "numpy",
                            4: "pandas",
                            5: "scipy",
                            6: "scikit-learn",
                            7: "matplotlib",
                            8: "jupyter"
                        }
                        if package_map[choice] not in packages_to_install:
                            packages_to_install.append(package_map[choice])
                            print(f"Added {package_map[choice]} to installation list")
                    else:
                        print("Invalid choice. Please enter a number between 1 and 8.")
                except ValueError:
                    print("Please enter a valid number.")

            if packages_to_install:
                print("\nInstalling selected packages...")
                if project_dir:
                    install_cmd = f"conda install -y -p {env_path} {' '.join(packages_to_install)}"
                else:
                    install_cmd = f"conda install -y -n {env_name} {' '.join(packages_to_install)}"
                subprocess.run(install_cmd, shell=True, check=True, executable='/bin/bash')
            
            print("Successfully installed all packages")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            return False

    def verify_installation(self, env_name, project_dir=None):
        """Verify the installation."""
        print("Verifying installation...")
        
        # Create the verification script
        verify_script = """
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print("=== Environment Verification ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("=============================")
"""
        try:
            # Use conda activate instead of mamba
            if project_dir:
                env_path = os.path.expanduser(f"{project_dir}/{env_name}")
                activate_cmd = f"conda activate {env_path}"
            else:
                activate_cmd = f"conda activate {env_name}"

            # Run the verification script
            verify_cmd = f"{activate_cmd} && python -c '{verify_script}'"
            subprocess.run(verify_cmd, shell=True, check=True, executable='/bin/bash')
            
            print("Installation verified successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error verifying installation: {e}")
            return False

    def create_job_script(self, env_name, project_dir=None):
        """Create a sample Slurm job script."""
        print("Creating sample job script...")
        
        script_content = """#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

module purge
eval "$(conda shell.bash hook)"
python your_script.py
"""
        
        if project_dir:
            script_content += f"conda activate {project_dir}/{env_name}\n"
        else:
            script_content += f"conda activate {env_name}\n"
        
        script_content += """
# Your Python script or command here
python your_script.py
"""
        
        script_path = f"run_{env_name}.job"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Created sample job script: {script_path}")
        return True

    def show_first_time_instructions(self):
        """Show additional instructions for first-time users."""
        print("\n=== First Time Setup Instructions ===")
        print("1. After this script completes, run:")
        print("   source ~/.bashrc")
        print("\n2. To activate your environment:")
        if self.project_dir:
            print(f"   mamba activate {self.project_dir}/{self.env_name}")
        else:
            print(f"   mamba activate {self.env_name}")
        print("\n3. To test your installation:")
        print("   python -c 'import torch; print(torch.__version__)'")
        print("\n4. For more information, visit:")
        print("   https://www.carc.usc.edu/user-guides/hpc-systems/software/conda")
        print("=====================================")

    def check_conda_setup(self):
        """Check if conda is already set up."""
        try:
            # Check if conda is available
            result = subprocess.run("conda --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("Conda is already installed and configured.")
                return True
            return False
        except:
            return False

    def run_interactive_setup(self):
        """Run the interactive setup process."""
        if self.is_first_time:
            self.show_first_time_welcome()
        else:
            self.show_welcome()

        # Check if conda is already set up
        if self.check_conda_setup():
            print("\nConda is already set up. Skipping initialization steps.")
            # Get user input for environment creation
            self.env_name = self.get_user_input("Enter environment name:", "pytorch_env", required=True)
            self.python_version = self.get_user_input("Enter Python version:", "3.10", required=True)
            self.project_dir = self.get_user_input("Enter project directory (optional):", "")

            print("\nStarting environment creation...\n")

            # Run only environment creation steps
            steps = [
                (lambda: self.create_conda_env(self.env_name, self.python_version, self.project_dir), "Creating conda environment"),
                (lambda: self.install_packages(self.env_name, self.project_dir), "Installing packages")
            ]
        else:
            # Get user input
            self.env_name = self.get_user_input("Enter environment name:", "pytorch_env", required=True)
            self.python_version = self.get_user_input("Enter Python version:", "3.10", required=True)
            self.project_dir = self.get_user_input("Enter project directory (optional):", "")

            print("\nStarting setup process...\n")

            # Run all setup steps
            steps = [
                (self.check_modules, "Loading conda module"),
                (self.initialize_mamba, "Initializing Mamba"),
                (lambda: self.create_conda_env(self.env_name, self.python_version, self.project_dir), "Creating conda environment"),
                (lambda: self.install_packages(self.env_name, self.project_dir), "Installing packages")
            ]

        for i, (step_func, step_name) in enumerate(steps, 1):
            print(f"\nStep {i}/{len(steps)}: {step_name}")
            if not step_func():
                print("\nSetup failed. Please check the errors above.")
                return False
            time.sleep(1)  # Give user time to read the output

        if self.is_first_time:
            self.show_first_time_instructions()
            self.save_config()
        else:
            print("\nSetup completed successfully!")
            print("\nTo activate the environment:")
            if self.project_dir:
                print(f"conda activate {self.project_dir}/{self.env_name}")
            else:
                print(f"conda activate {self.env_name}")
        
        print("\nTo clean up conda cache and free up space:")
        print("conda clean --all")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='CARC Conda Environment Setup')
    parser.add_argument('--env-name', help='Name of the conda environment')
    parser.add_argument('--python-version', help='Python version to use')
    parser.add_argument('--project-dir', help='Project directory path (optional)')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')
    
    args = parser.parse_args()

    if args.non_interactive:
        # Original non-interactive mode
        setup = InteractiveSetup()
        setup.check_modules()
        setup.initialize_mamba()
        setup.create_conda_env(args.env_name or "pytorch_env", args.python_version or "3.10", args.project_dir)
        setup.install_packages(args.env_name or "pytorch_env", args.project_dir)
    else:
        # Interactive mode
        setup = InteractiveSetup()
        setup.run_interactive_setup()

if __name__ == "__main__":
    main() 