import os

# logic for retrieving assets
def asset_abspath(resource_path):
    envs_path = os.path.dirname(__file__)
    asset_folder_path = os.path.join(envs_path, 'assets')
    return os.path.join(asset_folder_path, resource_path)
