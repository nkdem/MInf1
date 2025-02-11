# experiment1.py
import os
from base_experiment import BaseExperiment
from hear_ds import HEARDS
from train import AdamEarlyStopTrainer
from constants import MODELS

class FullAdam(BaseExperiment):
    def __init__(self, dataset, num_epochs=1, batch_size=16, 
                 number_of_experiments=1, learning_rates=None, cuda=False):
        super().__init__(dataset, cuda)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.number_of_experiments = number_of_experiments
        self.learning_rates = learning_rates
        self.experiment_name = f"full_adam_{num_epochs}epochs_{batch_size}batchsize_full"

    def run(self):
        # Training phase
        print(f"Starting Experiment 1 with {self.number_of_experiments} runs...")
        print(f"Parameters: epochs={self.num_epochs}, batch_size={self.batch_size}")
        print(f"Learning rates: {self.learning_rates}")

        # Run training for each experiment
        for i in range(1, self.number_of_experiments + 1):
            print(f"\nStarting experiment run {i}/{self.number_of_experiments}")
            base_dir = self.create_experiment_dir(self.experiment_name, i)
            adam = AdamEarlyStopTrainer(
                root_dir=self.dataset.root_dir,
                dataset=self.dataset,
                base_dir=base_dir,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                cuda=self.cuda,
            )
            adam.train()

        print("\nTraining phase completed. Starting results collection and analysis...")

        # Initialize results containers
        results = self.initialize_result_containers()
        output_dir = f'models/{self.experiment_name}/combined_results'
        os.makedirs(output_dir, exist_ok=True)

        # Collect and validate data consistency across experiments
        for i in range(1, 3):
            print(f"Collecting results from experiment run {i}...")
            base_dir = f'models/{self.experiment_name}/{i}'
            
            # Collect training and testing data
            training_data, testing_data = self.collect_experiment_data(base_dir)
            
            # Validate data consistency
            try:
                self.validate_data_consistency(training_data, testing_data)
            except AssertionError as e:
                print(f"Warning: Data consistency check failed for run {i}: {e}")
                continue

            # Collect results for each model
            for model in MODELS.keys():
                self.collect_model_results(base_dir, model, results)

        print("\nResults collection completed. Generating visualization plots...")

        # Generate all plots
        first_exp_dir = f'models/{self.experiment_name}/1'  # Use first experiment for reference
        self.generate_plots(
            results=results,
            base_dir=first_exp_dir,
            output_dir=output_dir,
            experiment_name=self.experiment_name
        )

        # Print summary statistics
        print("\nExperiment Summary:")
        for model in MODELS.keys():
            if results['accuracies'][model]:
                mean_acc = sum(results['accuracies'][model]) / len(results['accuracies'][model])
                mean_time = sum(results['training_times'][model]) / len(results['training_times'][model])
                print(f"\n{model}:")
                print(f"  Average Accuracy: {mean_acc:.2f}%")
                print(f"  Average Training Time: {mean_time:.2f} minutes")
                if results['losses'][model]:
                    final_losses = [losses[-1] for losses in results['losses'][model]]
                    mean_final_loss = sum(final_losses) / len(final_losses)
                    print(f"  Average Final Loss: {mean_final_loss:.4f}")

        print(f"\nExperiment completed. Results saved in: {output_dir}")

        # --- New part: Compare SNR performance ---
        # print("\nStarting SNR performance analysis (PESQ and STOI) for each environment...")
        # snr_results = self.compare_snr_performance()
        # print("\nSNR performance analysis completed.")

    def __str__(self):
        """String representation of the experiment configuration"""
        return (f"Experiment1("
                f"num_epochs={self.num_epochs}, "
                f"batch_size={self.batch_size}, "
                f"number_of_experiments={self.number_of_experiments}, "
                f"learning_rates={self.learning_rates}, "
                f"cuda={self.cuda})")

    def get_experiment_config(self):
        """Return experiment configuration as a dictionary"""
        return {
            "experiment_type": "Experiment1",
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "number_of_experiments": self.number_of_experiments,
            "learning_rates": self.learning_rates,
            "cuda": self.cuda,
            "experiment_name": self.experiment_name
        }

if __name__ == '__main__':
    # root_dir = '/Users/nkdem/Downloads/HEAR-DS'
    root_dir = '/home/s2203859/HEAR-DS'
    dataset = HEARDS(root_dir=root_dir, cuda=False, augmentation=False)
    experiment = FullAdam(
        dataset=dataset,
        num_epochs=240, 
        batch_size=16, 
        number_of_experiments=3, 
        cuda=True
    )
    experiment.run()
    print(experiment)
    print(experiment.get_experiment_config())
