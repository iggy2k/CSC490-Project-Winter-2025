import os
import subprocess
import yaml
from tqdm.notebook import tqdm
import sys

def install_dependencies(env_yaml_path="./environment.yaml"):
    """Installs required dependencies from an environment YAML file."""
    if "google.colab" in sys.modules:
      env_yaml_path="./PROJECT/environment.yaml"
    try:
        with open(env_yaml_path) as file_handle:
            environment_data = yaml.safe_load(file_handle)
        _install_pip_package('numpy==1.26.4')
        for dependency in tqdm(environment_data["dependencies"], total=len(environment_data["dependencies"])):
            if isinstance(dependency, dict):  # If it's a pip-specific dependency list
                for lib in dependency['pip']:
                    _install_pip_package(lib.split("=")[0])
            else:
                # Fixes ImportError: numpy.core.multiarray failed to import
                if dependency.split("=")[0] == 'numpy':
                    pass
                else:
                    _install_pip_package(dependency.split("=")[0])

        # Install additional required packages
        _install_pip_package("pycountry global_land_mask")

    except Exception as e:
        print(f"Error reading {env_yaml_path}: {e}")

def _install_pip_package(package_name):
    """Helper function to install a single package and handle errors."""
    try:
        subprocess.check_output(f'pip install {package_name}', shell=True)
        print(f"✅ Successfully installed {package_name}")
    except subprocess.CalledProcessError as err:
        print(f"⚠️ Error installing {package_name}: {err}")

def extract_dataset():
    """Extracts dataset from Google Drive and ensures necessary folders exist."""
    if not os.path.exists("490data.tar.gz"):
        os.system("cp /content/drive/My\\ Drive/490data.tar.gz .")

    if not os.path.exists("./data"):
        os.system("rm -rf ./data")
        os.makedirs("data")

    os.system("tar -xvzf 490data.tar.gz -C .")
    os.system("rm 490data.tar.gz")
    print("✅ Dataset extracted successfully.")

