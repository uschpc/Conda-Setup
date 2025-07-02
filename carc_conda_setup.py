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
        print("3. Install PyTorch and common data science packages\n")
        
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
            module_cmd = "module purge && module load conda"
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

    def create_conda_env(self, env_name, python_version):
        """Create a new conda environment."""
        print(f"Creating conda environment '{env_name}'...")
        try:
            subprocess.run(
                ["mamba", "create", "--name", env_name, f"python={python_version}", "-y"],
                check=True
            )
            print(f"Successfully created environment '{env_name}'")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating conda environment: {e}")
            return False

    def install_packages(self, env_name):
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

        try:
            # Create a new environment first
            create_cmd = f"conda create -y -n {env_name} python={self.python_version}"
            print(f"Creating new environment: {env_name}")
            subprocess.run(create_cmd, shell=True, check=True, executable='/bin/bash')

            # Ask about packages to install
            print("\nWould you like to install any of these packages?")
            print("Available packages:")
            print("1. PyTorch (for deep learning)")
            print("2. pandas (for data analysis)")
            print("3. scikit-learn (for machine learning)")
            print("4. matplotlib (for plotting)")
            
            packages_to_install = []
            while True:
                choice = self.get_user_input("\nEnter package number to install (or press Enter to finish):", "")
                if not choice:
                    break
                try:
                    choice = int(choice)
                    if 1 <= choice <= 4:
                        package_map = {
                            1: "pytorch",
                            2: "pandas",
                            3: "scikit-learn",
                            4: "matplotlib"
                        }
                        if package_map[choice] not in packages_to_install:
                            packages_to_install.append(package_map[choice])
                            print(f"Added {package_map[choice]} to installation list")
                    else:
                        print("Invalid choice. Please enter a number between 1 and 4.")
                except ValueError:
                    print("Please enter a valid number.")

            if packages_to_install:
                print("\nInstalling selected packages...")
                
                # Activate the environment first
                activate_cmd = f"conda activate {env_name}"
                
                # Install packages
                for package in packages_to_install:
                    if package == "pytorch":
                        print("Installing PyTorch...")
                        install_cmd = f"{activate_cmd} && pip3 install torch torchvision torchaudio"
                    else:
                        install_cmd = f"{activate_cmd} && conda install -y {package}"
                    
                    subprocess.run(install_cmd, shell=True, check=True, executable='/bin/bash')
            
            print("Successfully installed all packages")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            return False

    def verify_installation(self, env_name):
        """Verify the installation."""
        print("Verifying installation...")
        
        # Create the verification script
        verify_script = """
import sys
print("=== Environment Verification ===")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not installed")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy not installed")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError:
    print("Pandas not installed")

try:
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("Scikit-learn not installed")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError:
    print("Matplotlib not installed")

print("=============================")
"""
        try:
            # Use conda activate instead of mamba
            activate_cmd = f"conda activate {env_name}"

            # Run the verification script
            verify_cmd = f"{activate_cmd} && python -c '{verify_script}'"
            subprocess.run(verify_cmd, shell=True, check=True, executable='/bin/bash')
            
            print("Installation verified successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error verifying installation: {e}")
            return False

    def show_first_time_instructions(self):
        """Show additional instructions for first-time users."""
        print("\n=== First Time Setup Instructions ===")
        print("1. After this script completes, run:")
        print("   source ~/.bashrc")
        print("\n2. To activate your environment:")
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

    def check_bash_configs(self):
        """Check if user's bash configuration files exist and copy from /etc/skel if they don't."""
        home_dir = os.path.expanduser("~")
        bash_profile = os.path.join(home_dir, ".bash_profile")
        bashrc = os.path.join(home_dir, ".bashrc")
        
        print("\nChecking bash configuration files...")
        
        # Check .bash_profile
        print(f"\nChecking .bash_profile at: {bash_profile}")
        if os.path.exists(bash_profile):
            print("✓ .bash_profile exists")
        else:
            print("✗ .bash_profile not found")
            print("Creating .bash_profile...")
            try:
                shutil.copy("/etc/skel/.bash_profile", bash_profile)
                print("✓ Successfully created .bash_profile")
            except Exception as e:
                print(f"✗ Error creating .bash_profile: {e}")
                return False
        
        # Check .bashrc
        print(f"\nChecking .bashrc at: {bashrc}")
        if os.path.exists(bashrc):
            print("✓ .bashrc exists")
        else:
            print("✗ .bashrc not found")
            print("Creating .bashrc...")
            try:
                shutil.copy("/etc/skel/.bashrc", bashrc)
                print("✓ Successfully created .bashrc")
            except Exception as e:
                print(f"✗ Error creating .bashrc: {e}")
                return False
        
        print("\n✓ Bash configuration check completed successfully")
        return True

    def run_interactive_setup(self):
        """Run the interactive setup process."""
        # Check if this is first time
        if not self.is_first_time:
            print("This script is only for first-time users.")
            print("If you need to create a new conda environment, please use conda directly.")
            return False

        # Step 1: Check and set up bash configuration files
        print("\nStep 1: Checking bash configuration files...")
        if not self.check_bash_configs():
            print("Failed to set up bash configuration files. Please contact system administrator.")
            return False

        self.show_first_time_welcome()

        # Step 2: First time initialization
        print("\nStep 2: Initializing conda environment...")
        try:
            # Module purge and load conda
            print("Loading conda module...")
            subprocess.run("module purge && module load conda", shell=True, check=True, executable='/bin/bash')
            
            # Initialize mamba
            print("Initializing Mamba...")
            subprocess.run("mamba init bash", shell=True, check=True, executable='/bin/bash')
            
            # Source bashrc
            print("Sourcing ~/.bashrc...")
            subprocess.run("source ~/.bashrc", shell=True, check=True, executable='/bin/bash')
            
            print("✓ Conda environment initialized successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error during initialization: {e}")
            return False

        # Step 3: Get user input and create environment
        print("\nStep 3: Creating Python environment...")
        self.env_name = self.get_user_input("Enter environment name:", "pytorch_env", required=True)
        self.python_version = self.get_user_input("Enter Python version:", "3.10", required=True)

        # Create conda environment
        create_cmd = f"conda create -y -n {self.env_name} python={self.python_version}"
        print(f"Creating new environment: {self.env_name}")
        try:
            subprocess.run(create_cmd, shell=True, check=True, executable='/bin/bash')
            print("✓ Environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error creating environment: {e}")
            return False

        # Step 4: Install packages
        print("\nStep 4: Installing packages...")
        # Ask about packages to install
        print("\nWould you like to install any of these packages?")
        print("Available packages:")
        print("1. PyTorch (for deep learning)")
        print("2. pandas (for data analysis)")
        print("3. scikit-learn (for machine learning)")
        print("4. matplotlib (for plotting)")
        
        packages_to_install = []
        while True:
            choice = self.get_user_input("\nEnter package number to install (or press Enter to finish):", "")
            if not choice:
                break
            try:
                choice = int(choice)
                if 1 <= choice <= 4:
                    package_map = {
                        1: "pytorch",
                        2: "pandas",
                        3: "scikit-learn",
                        4: "matplotlib"
                    }
                    if package_map[choice] not in packages_to_install:
                        packages_to_install.append(package_map[choice])
                        print(f"Added {package_map[choice]} to installation list")
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")

        if packages_to_install:
            print("\nInstalling selected packages...")
            # Initialize conda and activate the environment
            init_cmd = "eval \"$(conda shell.bash hook)\""
            activate_cmd = f"conda activate {self.env_name}"
            
            # Install packages
            for package in packages_to_install:
                if package == "pytorch":
                    print("Installing PyTorch...")
                    install_cmd = f"{init_cmd} && {activate_cmd} && pip3 install torch torchvision torchaudio"
                else:
                    install_cmd = f"{init_cmd} && {activate_cmd} && conda install -y {package}"
                
                try:
                    subprocess.run(install_cmd, shell=True, check=True, executable='/bin/bash')
                    print(f"✓ Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Error installing {package}: {e}")
                    return False

        self.show_first_time_instructions()
        self.save_config()
        
        print("\nTo clean up conda cache and free up space:")
        print("conda clean --all")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='CARC Conda Environment Setup (First-time users only)')
    parser.add_argument('--env-name', help='Name of the conda environment')
    parser.add_argument('--python-version', help='Python version to use')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')
    
    args = parser.parse_args()

    setup = InteractiveSetup()
    
    if not setup.is_first_time:
        print("This script is only for first-time users.")
        print("If you need to create a new conda environment, please use conda directly.")
        sys.exit(1)

    if args.non_interactive:
        # Non-interactive mode
        if not setup.check_bash_configs():
            print("Failed to set up bash configuration files. Please contact system administrator.")
            sys.exit(1)
            
        # Initialize conda environment
        try:
            subprocess.run("module purge && module load conda", shell=True, check=True, executable='/bin/bash')
            subprocess.run("mamba init bash", shell=True, check=True, executable='/bin/bash')
            subprocess.run("source ~/.bashrc", shell=True, check=True, executable='/bin/bash')
        except subprocess.CalledProcessError as e:
            print(f"Error during initialization: {e}")
            sys.exit(1)
            
        setup.create_conda_env(args.env_name or "pytorch_env", args.python_version or "3.10")
        setup.install_packages(args.env_name or "pytorch_env")
    else:
        # Interactive mode
        setup.run_interactive_setup()

if __name__ == "__main__":
    main() 