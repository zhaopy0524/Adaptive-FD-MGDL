import json
import os


def save(output_folder, args, results, loss_data):

    # Create folder if it doesn't exist
    output_folder = os.path.join(os.getcwd(), output_folder)
    if os.path.exists(output_folder):
        raise FileExistsError(f"The folder '{output_folder}' already exists. Please specify a new folder.")
    else:
        os.makedirs(output_folder)

    # Save command-line arguments to JSON
    config_path = os.path.join(output_folder, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    results_path = os.path.join(output_folder, 'results.json')
    print('saving to ', results_path)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    losses_path = os.path.join(output_folder, 'losses.json')
    with open(losses_path, 'w') as f:
        json.dump(loss_data, f)