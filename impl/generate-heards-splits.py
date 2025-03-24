from itertools import groupby
import random
from heards_dataset import BackgroundDataset, DuplicatedMixedAudioDataset, MixedAudioDataset, SpeechDataset, split_background_dataset
from helpers import get_truly_random_seed_through_os, seed_everything
import os
import pickle
from torch.utils.data import ConcatDataset, Dataset
import soundfile as sf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--heards_dir', type=str, default='/Volumes/SSD/Datasets/HEAR-DS')
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

        # Create random SNR dataset first
        print("Creating random SNR dataset...")
        train_mixed_ds_random = MixedAudioDataset(background_dataset=train_background_speech_ds, speech_dataset=full_speech_ds, fixed_snr=False)
        test_mixed_ds_random = MixedAudioDataset(background_dataset=test_background_speech_ds, speech_dataset=full_speech_ds, fixed_snr=False)


        # Create fixed SNR dataset using the speech cache from random SNR dataset
        print("Creating fixed SNR dataset (copying speech cache from random SNR dataset)")
        train_mixed_ds_fixed = MixedAudioDataset(background_dataset=train_background_speech_ds, speech_dataset=full_speech_ds, fixed_snr=True, speech_cache=train_mixed_ds_random.speech_cache)
        test_mixed_ds_fixed = MixedAudioDataset(background_dataset=test_background_speech_ds, speech_dataset=full_speech_ds, fixed_snr=True, speech_cache=test_mixed_ds_random.speech_cache)

        # For random SNR, use regular background datasets
        train_combined_random = ConcatDataset([train_background_ds, train_mixed_ds_random])
        test_combined_random = ConcatDataset([test_background_ds, test_mixed_ds_random])

        # count number of classes 
        train_classes = {'random_snr': {}, 'fixed_snr': {}}
        for env, samples in groupby(train_combined_random, key=lambda x: x[2]):
            train_classes['random_snr'][env] = len(list(samples))
        test_classes = {'random_snr': {}, 'fixed_snr': {}}
        for env, samples in groupby(test_combined_random, key=lambda x: x[2]):
            test_classes['random_snr'][env] = len(list(samples))
        

        # For fixed SNR, duplicate background samples for each SNR level
        train_mixed_duplicated = DuplicatedMixedAudioDataset(train_mixed_ds_fixed)
        test_mixed_duplicated = DuplicatedMixedAudioDataset(test_mixed_ds_fixed)
        train_combined_fixed = ConcatDataset([train_background_ds, train_mixed_duplicated])
        test_combined_fixed = ConcatDataset([test_background_ds, test_mixed_duplicated])

        # the only difference is that fixed_snr _speech environments * number of snr levels
        train_classes['fixed_snr'] = {env: count * len(train_mixed_ds_random.snr_levels) if env.startswith('SpeechIn_') else count for env, count in train_classes['random_snr'].items()}
        test_classes['fixed_snr'] = {env: count * len(test_mixed_ds_random.snr_levels) if env.startswith('SpeechIn_') else count for env, count in test_classes['random_snr'].items()}

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'split_{i}.pkl'), 'wb') as f:
            data = {
                'random_snr': {
                    'train': train_combined_random,
                    'test': test_combined_random,
                    'subsets': {
                        'train': {
                            'background': train_background_ds,
                            'speech': train_background_speech_ds,
                            'mixed': train_mixed_ds_random
                        },
                        'test': {
                            'background': test_background_ds,
                            'speech': test_background_speech_ds,
                            'mixed': test_mixed_ds_random
                        }
                    }
                },
                'fixed_snr': {
                    'train': train_combined_fixed,
                    'test': test_combined_fixed,
                    'subsets': {
                        'train': {
                            'background': train_mixed_duplicated,
                            'speech': train_background_speech_ds,
                            'mixed': train_mixed_ds_fixed
                        },
                        'test': {
                            'background': test_mixed_duplicated,
                            'speech': test_background_speech_ds,
                            'mixed': test_mixed_ds_fixed
                        }
                    }
                },
                'seed': seed_val,
                'classes': {
                    'random_snr': train_classes['random_snr'],
                    "train": {
                        "random_snr": train_classes['random_snr'],
                        "fixed_snr": train_classes['fixed_snr']
                    },
                    "test": {
                        "random_snr": test_classes['random_snr'],
                        "fixed_snr": test_classes['fixed_snr']
                    }
                }
            }
        pickle.dump(data, open(os.path.join(args.output_dir, f'split_{i}.pkl'), 'wb'))

        # os.makedirs(os.path.join(args.output_dir, f'split_{i}_samples'), exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, f'split_{i}_samples', 'random_snr'), exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, f'split_{i}_samples', 'fixed_snr'), exist_ok=True)

        # for split in ['train', 'test']:
        #     for env, samples in groupby(data['fixed_snr'][split], key=lambda x: x[2]):
        #         # pick 10 random samples
        #         subset = random.sample(list(samples), 10)


        #         base_name = sample[5][0]
        #         snr = sample[6]
        #         if snr is None:
        #             continue
        #         os.makedirs(os.path.join(args.output_dir, f'split_{i}_samples', 'fixed_snr', env, str(snr)), exist_ok=True)
        #         sf.write(os.path.join(args.output_dir, f'split_{i}_samples', 'fixed_snr', env, str(snr), f'{base_name}_L.wav'), sample[0][0], 16000)
        #         sf.write(os.path.join(args.output_dir, f'split_{i}_samples', 'fixed_snr', env, str(snr), f'{base_name}_R.wav'), sample[0][1], 16000)



        # as a sanity check, let's read the files back and compare against created
        with open(os.path.join(args.output_dir, f'split_{i}.pkl'), 'rb') as f:
            data = pickle.load(f)
            assert data['seed'] == seed_val
            
            # Check random SNR variant
            assert len(data['random_snr']['train']) == len(train_combined_random)
            assert len(data['random_snr']['test']) == len(test_combined_random)
            assert len(data['random_snr']['subsets']['train']['background']) == len(train_background_ds)
            assert len(data['random_snr']['subsets']['train']['speech']) == len(train_background_speech_ds)
            assert len(data['random_snr']['subsets']['train']['mixed']) == len(train_mixed_ds_random)
            assert len(data['random_snr']['subsets']['test']['background']) == len(test_background_ds)
            assert len(data['random_snr']['subsets']['test']['speech']) == len(test_background_speech_ds)
            assert len(data['random_snr']['subsets']['test']['mixed']) == len(test_mixed_ds_random)
            
            # Check fixed SNR variant
            assert len(data['fixed_snr']['train']) == len(train_combined_fixed)
            assert len(data['fixed_snr']['test']) == len(test_combined_fixed)
            assert len(data['fixed_snr']['subsets']['train']['background']) == len(train_mixed_duplicated)
            assert len(data['fixed_snr']['subsets']['train']['speech']) == len(train_background_speech_ds)
            assert len(data['fixed_snr']['subsets']['train']['mixed']) == len(train_mixed_ds_fixed)
            assert len(data['fixed_snr']['subsets']['test']['background']) == len(test_mixed_duplicated)
            assert len(data['fixed_snr']['subsets']['test']['speech']) == len(test_background_speech_ds)
            assert len(data['fixed_snr']['subsets']['test']['mixed']) == len(test_mixed_ds_fixed)