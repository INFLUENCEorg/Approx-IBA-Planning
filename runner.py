
import argparse
import pandas as pd
import yaml
import os
from environments.MarsRover.Test import run_MR
from environments.TrafficGrid.Test import run_TG
from environments.FireFighters.Test import run_FF


def main(parameters):

    env_type = parameters['env_type']
    parameters['path']= generate_path(parameters)

    if env_type=='MR':
        run_MR(parameters)
    elif env_type=='TG':
        run_TG(parameters)
    elif env_type=='FF':
        run_FF(parameters)


def generate_path(parameters):
        path = parameters['name']
        result_path = os.path.join("results", path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(result_path+'/config.yml', 'w') as outfile:
            yaml.dump(parameters, outfile, default_flow_style=False)
        return result_path


def get_config_file():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', default=None, help='config file')
    args = parser.parse_args()
    return args.config


def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']


if __name__ == "__main__":
    config_file = get_config_file()
    parameters = read_parameters(config_file)
    main(parameters)




