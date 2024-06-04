import subprocess

# Loop through bands from 0 to 4 and execute the command
for feature in ['dasm', 'rasm', 'dcau']:
    for band in range(5):
        command = f'python -u "/home/wf/EEG_GTN/main.py" --dataset seed_adj --train_mode si --batch_size 16 --bands {band} --feature_type {feature}'
        print(f"Executing: {command}")
        subprocess.run(command, shell=True)
