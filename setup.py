#!/usr/bin/env python3
"""Setup for basalt Python bindings.

The C++ extension is pre-built via CMake. This setup.py packages it for distribution.

Quick start:
    1. Build basalt with CMake:
       cmake --preset relwithdebinfo
       cmake --build build/relwithdebinfo

    2. Install in development mode:
       pip install -e .
       
    Or with a custom build directory:
       CMAKE_BINARY_DIR=/path/to/build pip install -e .

The script will automatically find and install the pre-built .so file.
"""

import glob
import os
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
    """Custom build_py that copies pre-built basalt extension to build/lib."""
    
    def run(self):
        """Copy pre-built basalt extension."""
        # Get CMake build directory
        cmake_build = os.environ.get(
            "CMAKE_BINARY_DIR",
            str(Path(__file__).parent / "build" / "relwithdebinfo")
        )
        
        python_build_dir = Path(cmake_build) / "python"
        
        # Find the basalt extension
        so_files = list(python_build_dir.glob("basalt*.so"))
        
        if not so_files:
            raise FileNotFoundError(
                f"Could not find basalt extension in {python_build_dir}\n"
                f"Make sure to build basalt with CMake first:\n"
                f"  cd {Path(__file__).parent}\n"
                f"  cmake --preset relwithdebinfo\n"
                f"  cmake --build build/relwithdebinfo"
            )
        
        basalt_so = so_files[0]
        print(f"✓ Using basalt extension: {basalt_so}")
        
        # Copy to build/lib directory (will be installed to site-packages)
        build_lib = Path(self.build_lib)
        build_lib.mkdir(parents=True, exist_ok=True)
        
        dest = build_lib / basalt_so.name
        print(f"✓ Copying to: {dest}")
        shutil.copy2(basalt_so, dest)
        
        # Also copy as basalt.so for compatibility
        if "cpython" in basalt_so.name or basalt_so.name != "basalt.so":
            generic_dest = build_lib / "basalt.so"
            if generic_dest != dest:
                shutil.copy2(basalt_so, generic_dest)
                print(f"✓ Also copied as: {generic_dest}")
        
        # Call parent run (will copy any .py files)
        build_py.run(self)


setup(
    name="basalt",
    version="0.0.1",
    description="Basalt: Visual-inertial SLAM",
    long_description=__doc__,
    author="Basalt Contributors",
    url="https://gitlab.com/VladyslavUsenko/basalt.git",
    packages=[],  # No packages, just the .so file
    py_modules=[],  # No pure Python modules
    cmdclass={
        "build_py": BuildPyCommand,
    },
    python_requires=">=3.8",
    zip_safe=False,
)
