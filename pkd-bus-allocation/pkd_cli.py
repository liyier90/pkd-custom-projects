import sys

from pathlib import Path

NODE_TYPES = ["dabble", "draw", "input", "model", "output"]
YES_OR_NO = ["y", "n"]

PYTHON_BOILERPLATE = """
# AISG Boilerplate
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}
"""

YAML_BOILERPLATE = """
input: []
output: []
"""

def format_boilerplate(boilerplate):
    return boilerplate[1:]

def format_msg(msg, default=None, choices=None):
    if default:
        return f"{msg} [{default}]: "
    elif choices:
        return f"{msg} ({'/'.join(choices)}): "
    else:
        return f"{msg}: "


def main(argv):
    if len(argv):
        mode = argv[0]
        if mode == "create_node":
            print("Create new node")
            project_dir = Path.cwd()

            # Node directory
            node_dir = "src/custom_nodes"
            user_input = input(
                format_msg(
                    f"Enter node directory relative to {project_dir}", default=node_dir
                )
            )
            if user_input:
                node_dir = user_input
            node_dir = project_dir / node_dir

            # Node type
            user_input = input(format_msg("Select node type", choices=NODE_TYPES))
            while user_input not in NODE_TYPES:
                user_input = input(
                    format_msg("Invalid type! Select node type", choices=NODE_TYPES)
                )
            node_type = user_input

            # Node name
            node_name = "my_custom_node"
            user_input = input(format_msg("Enter node name", default=node_name))
            while (node_dir / node_type / f"{user_input}.py").exists() or (
                node_dir / node_type / f"{node_name}.py"
            ).exists():
                user_input = input(
                    format_msg(
                        "Node already exists! Enter node name", default=node_name
                    )
                )
            if user_input:
                node_name = user_input

            config_path = node_dir / "configs" / node_type / f"{node_name}.yml"
            script_path = node_dir / node_type / f"{node_name}.py"
            print(f"\nNode directory:\t{node_dir}")
            print(f"Node type:\t{node_type}")
            print(f"Node name:\t{node_name}")
            print("\nCreating the following files:")
            print(f"\tConfig file: {config_path}")
            print(f"\tScript file: {script_path}")

            user_input = input(format_msg("Proceed?", choices=YES_OR_NO)).lower()
            while user_input not in YES_OR_NO:
                user_input = input(
                    format_msg("Invalid choice! Proceed?", choices=YES_OR_NO)
                ).lower()
            if user_input == "y":
                config_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, "w") as outfile:
                    outfile.write(format_boilerplate(YAML_BOILERPLATE))
                with open(script_path, "w") as outfile:
                    outfile.write(format_boilerplate(PYTHON_BOILERPLATE))
                print("Created node!")
            else:
                print("Aborted!")


if __name__ == "__main__":
    main(sys.argv[1:])
