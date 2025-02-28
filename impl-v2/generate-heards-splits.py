from heards_dataset import BackgroundDataset, MixedAudioDataset, SpeechDataset, split_background_dataset
from helpers import get_truly_random_seed_through_os, seed_everything
import os
import pickle
from torch.utils.data import ConcatDataset



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--heards_dir', type=str, default='/Users/nkdem/Downloads/HEAR-DS')
    parser.add_argument('--chime_dir', type=str, default='/Volumes/SSD/Datasets/CHiME3/CHiME3-Isolated-DEV/dt05_bth')
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='splits')

    args = parser.parse_args()
    for i in range(args.num_splits):
        seed_val = get_truly_random_seed_through_os()
        seed_everything(seed_val)
        print(f"Seed: {seed_val}")

        full_background_ds = BackgroundDataset(root_dir=args.heards_dir)
        full_speech_ds = SpeechDataset(chime_dir=args.chime_dir)

        train_background_ds, test_background_ds, train_background_speech_ds, test_background_speech_ds = split_background_dataset(full_background_ds)

        train_mixed_ds = MixedAudioDataset(background_dataset=train_background_speech_ds, speech_dataset=full_speech_ds)
        test_mixed_ds = MixedAudioDataset(background_dataset=test_background_speech_ds, speech_dataset=full_speech_ds)

        train_combined = ConcatDataset([train_background_ds, train_mixed_ds])
        test_combined = ConcatDataset([test_background_ds, test_mixed_ds])
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'split_{i}.pkl'), 'wb') as f:
            pickle.dump({
                'train': train_combined,
                'test': test_combined,
                'seed': seed_val,
                'subsets': {
                    'train': {
                        'background': train_background_ds,
                        'speech': train_background_speech_ds,
                        'mixed': train_mixed_ds
                    },
                    'test': {
                        'background': test_background_ds,
                        'speech': test_background_speech_ds,
                        'mixed': test_mixed_ds  
                    }
                }
            }, f)
        
        # as a sanity check, let's read the fiels back and compare against created
        with open(os.path.join(args.output_dir, f'split_{i}.pkl'), 'rb') as f:
            data = pickle.load(f)
            assert len(data['train']) == len(train_combined)
            assert len(data['test']) == len(test_combined)
            assert data['seed'] == seed_val
            assert len(data['subsets']['train']['background']) == len(train_background_ds)
            assert len(data['subsets']['train']['speech']) == len(train_background_speech_ds)
            assert len(data['subsets']['train']['mixed']) == len(train_mixed_ds)
            assert len(data['subsets']['test']['background']) == len(test_background_ds)
            assert len(data['subsets']['test']['speech']) == len(test_background_speech_ds)
            assert len(data['subsets']['test']['mixed']) == len(test_mixed_ds)