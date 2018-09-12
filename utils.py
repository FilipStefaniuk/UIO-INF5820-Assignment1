import argparse
import json


def parse_args():
    """Parse comand line arguments.

    # Returns
        parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', required=True, help='path to data file')
    parser.add_argument('-o', '--output', required=False, default='./experiments/test', help='output directory')
    parser.add_argument('-c', '--config', required=False, help='path to json config file')
    parser.add_argument('-t', '--test', action='store_true', help='evaluate model on test dataset')

    return parser.parse_args()


def parse_config(path):
    """Parse config file.

    # Arguments:
        path: path to config file.

    # Returns:
        dictionary with config values.
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_results(path, data):
    """Saves data into json file.

    # Arguments:
        path: path to json file.
        data: data to save.
    """
    with open(path, 'w') as f:
        json.dump(data, f)
