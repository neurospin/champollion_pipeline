#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScriptFactory: A base class for creating flexible command-line scripts.
"""

import os
import sys
import pprint
import argparse
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from os.path import dirname, join, exists
from subprocess import run, check_call

class ScriptFactory(ABC):
    """Base class for creating flexible command-line scripts."""

    def __init__(self, script_name: str, description: str):
        """
        Initialize the script factory.

        Args:
            script_name: Name of the script (used in help messages).
            description: Description of the script.
        """
        self.script_name = script_name
        self.description = description
        self.parser = self._create_parser()
        self.args = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        return argparse.ArgumentParser(
            prog=self.script_name,
            description=self.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    def add_argument(self, *args, **kwargs) -> None:
        """Add an argument to the parser."""
        self.parser.add_argument(*args, **kwargs)

    def add_required_argument(self, name: str, help_text: str, type_: type = str) -> None:
        """Add a required argument."""
        self.add_argument(name, type=type_, required=True, help=help_text)

    def add_optional_argument(
        self,
        name: str,
        help_text: str,
        default: Any = None,
        type_: type = str,
        action: Optional[str] = None
    ) -> None:
        """Add an optional argument."""
        if action:
            self.add_argument(name, action=action, help=help_text)
        else:
            self.add_argument(name, type=type_, default=default, help=help_text)

    def add_flag(self, name: str, help_text: str) -> None:
        """Add a boolean flag."""
        self.add_optional_argument(name, help_text, action="store_true")

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        self.args = self.parser.parse_args(args)
        return self.args

    def validate_paths(self, paths: List[str]) -> bool:
        """Validate that all paths exist."""
        for path in paths:
            if not exists(path):
                print(f"Error: Path does not exist: {path}")
                return False
        return True

    def print_args(self) -> None:
        """Print the parsed arguments."""
        print(f"{self.script_name} arguments:")
        pprint.pprint(vars(self.args))

    def build_command(
        self,
        script_path: str,
        required_args: List[str],
        optional_args: Dict[str, Any],
        defaults: Dict[str, Any]
    ) -> List[str]:
        """
        Build a command to execute another script.

        Args:
            script_path: Path to the script to execute.
            required_args: List of required argument names.
            optional_args: Dictionary of optional argument names and their values.
            defaults: Dictionary of default values for optional arguments.

        Returns:
            List of command components.
        """
        cmd = [sys.executable, script_path]

        # Add required arguments
        for arg in required_args:
            value = getattr(self.args, arg)
            cmd.append(f"--{arg}={value}")

        # Add optional arguments if they differ from defaults
        for arg, default in defaults.items():
            value = getattr(self.args, arg)
            if value != default:
                if isinstance(value, bool) and value:
                    cmd.append(f"--{arg}")
                elif isinstance(value, list):
                    for item in value:
                        cmd.append(f"--{arg}={item}")
                else:
                    cmd.append(f"--{arg}={value}")

        return cmd

    def execute_command(self, cmd: List[str], shell: bool = False) -> int:
        """
        Execute a command.

        Args:
            cmd: Command to execute.
            shell: Whether to use shell execution.

        Returns:
            Return code of the command.
        """
        print("Running command:")
        print(" ".join(cmd))
        try:
            if shell:
                return run(" ".join(cmd), shell=True, executable="/bin/bash").returncode
            else:
                return check_call(cmd)
        except Exception as e:
            print(f"Error running command: {e}")
            return 1

    @abstractmethod
    def run(self) -> int:
        """Run the script. Must be implemented by subclasses."""
        pass

    def main(self) -> int:
        """Main entry point for the script."""
        self.parse_args()
        self.print_args()
        return self.run()
