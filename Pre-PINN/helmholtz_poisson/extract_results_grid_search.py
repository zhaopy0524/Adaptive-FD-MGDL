import os
import json

def scan_folder(root_folder):
    results = []
    for root, dirs, files in os.walk(root_folder):
        if 'results.json' in files and 'config.json' in files and 'losses.json' in files:
            with open(os.path.join(root, 'results.json'), 'r') as f:
                result_data = json.load(f)
            with open(os.path.join(root, 'config.json'), 'r') as f:
                config_data = json.load(f)
            with open(os.path.join(root, 'losses.json'), 'r') as f:
                loss_data = json.load(f)

            try:
                final_loss = result_data['final_loss']
            except KeyError:
                print(f"Warning: 'final_loss' not found in {os.path.join(root, 'results.json')}")
                continue

            try:
                min_loss = min(loss_data['loss'])
            except (KeyError, ValueError):
                print(f"Warning: Could not extract minimum loss from {os.path.join(root, 'losses.json')}")
                continue

            try:
                epochs = loss_data['epoch']
                losses = loss_data['loss']
            except KeyError:
                print(f"Warning: Could not extract epochs or losses from {os.path.join(root, 'losses.json')}")
                continue

            results.append({
                'folder': root,
                'final_loss': final_loss,
                'min_loss': min_loss,
                'epochs': epochs,
                'losses': losses,
                'config': config_data
            })

    return results

if __name__ == "__main__":
    root_folder = "results/grid_search"  # Replace with the root folder containing your experiment sub-folders
    results = scan_folder(root_folder)
    # Sort the experiments by min_loss
    results = sorted(results, key=lambda x: x['min_loss'], reverse=True)

    # Print all experiments from lowest to highest minimal loss
    print("All Experiments (sorted by minimal loss):")
    for res in results:
        print(f"Folder: {res['folder']}")
        print(f"Final Loss: {res['final_loss']}")
        print(f"Minimum Loss: {res['min_loss']}")
        print(f"Epochs: {len(res['epochs'])}")
        print(f"Config: {res['config']}")
        print('-' * 40)

    print(f'{len(results)} experiments found!')
