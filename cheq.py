import yaml

with open('params.yaml','r') as file:
    config_data = yaml.safe_load(file)

bounds = config_data['test_config']['list_of_holes']

print(bounds)