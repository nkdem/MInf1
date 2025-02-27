import os

# Define the path to the speech enhancement folder
speech_enhancement_folder = 'speech-enhancement'
classification_folder = 'classification'

# Find all experiment files in the speech enhancement folder
experiment_files = [os.path.join('speech-enhancement', f) for f in os.listdir(speech_enhancement_folder) if f.startswith('expr-') and os.path.isfile(os.path.join(speech_enhancement_folder, f))]
# now in classification folder
experiment_files += [os.path.join(classification_folder, 'HEAR-DS', f) for f in os.listdir('classification/HEAR-DS') if f.startswith('expr-') and os.path.isfile(os.path.join(classification_folder, 'HEAR-DS', f))]
experiment_files += [os.path.join(classification_folder, 'TUT', f) for f in os.listdir('classification/TUT') if f.startswith('expr-') and os.path.isfile(os.path.join(classification_folder, 'TUT', f))]

# Print the list of experiments
print("Available experiments in the speech enhancement folder:")
for i, experiment in enumerate(experiment_files, 1):
    print(f"{i}. {experiment}")

# Ask the user to choose an experiment
while True:
    try:
        choice = int(input("Enter the number of the experiment you want to run: "))
        if 1 <= choice <= len(experiment_files):
            selected_experiment = experiment_files[choice - 1]
            print(f"Running experiment: {selected_experiment}")

            # how many experiments to run
            num_experiments = int(input("How many experiments do you want to run? "))
            os.system(f"python {selected_experiment} --experiment_no {num_experiments}")
            # os.system(f"python {selected_experiment} --experiment_no {num_experiments} --cuda")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and", len(experiment_files))
    except ValueError:
        print("Invalid input. Please enter a number.")
