from setuptools import setup
import os

def create_env():
    """Create a .env file in the package root directory."""
    with open(".env", "w") as f:
        f.write(f'PROJECT_DIR = "{os.path.dirname(os.path.abspath(__file__))}"')
    print(".env file created!")

if __name__ == "__main__":
    create_env()
    setup()