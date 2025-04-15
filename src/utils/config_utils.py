import os
import yaml
import argparse
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with optional command line overrides.

    Args:
        config_path: Path to the configuration file. If None, uses default path.

    Returns:
        Dict containing the configuration.
    """
    if config_path is None or config_path == "":
        # Use default config path relative to the vehicle_sim directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, 'config', 'config.yaml')
        print(f"Using default config path: {config_path}")

    # Load the YAML configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vehicle Simulator')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', type=str2bool, help='Enable debug mode')
    parser.add_argument('--visualize', type=str2bool, help='Enable visualization')
    parser.add_argument('--visualize_hud', type=str2bool, help='Enable HUD visualization')
    parser.add_argument('--num_vehicles', type=int, help='Number of vehicles')
    parser.add_argument('--max_steps', type=int, help='Maximum steps per episode')

    args = parser.parse_args()

    # Override config with command line arguments if provided
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    if args.debug is not None:
        config['simulation']['debug_mode'] = args.debug

    if args.visualize is not None:
        config['simulation']['visualize'] = args.visualize

    if args.visualize_hud is not None:
        config['simulation']['visualize_hud'] = args.visualize_hud

    if args.num_vehicles is not None:
        config['simulation']['num_vehicles'] = args.num_vehicles

    if args.max_steps is not None:
        config['simulation']['max_steps'] = args.max_steps

    return config

def str2bool(v):
    """
    Convert string to boolean for argparse.

    Args:
        v: String value to convert.

    Returns:
        Boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')