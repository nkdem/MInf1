import os 

# os.system(f"python heards-speech-enh.py --experiment_no {5}")
for i in range(1, 6):
    os.system(f"python heards-speech-enh-one-model.py --experiment_no {i}")