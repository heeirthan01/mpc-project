import yaml
import numpy as np

with open('params.yaml','r') as file:
    config_data = yaml.safe_load(file)

#bounds = config_data['test_config']['list_of_holes']
cord = [27.8,45.3]
cord2 = [28,45]
print(np.linalg.norm(np.array(cord) - np.array(cord2)))