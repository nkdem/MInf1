import os

one_model = "speech-enhancement/expr_baseline.py"
per_env = "speech-enhancement/expr_per-env.py"
voicebank = "speech-enhancement/voicebank/baseline.py"
for i in range(5):
    for model in [one_model, per_env, voicebank]:
        os.system(f"python {model} --experiment_no {i+1}")
        print(f"Experiment {i+1} completed.")