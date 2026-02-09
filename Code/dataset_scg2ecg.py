import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class BCGECDataset(Dataset):
    def __init__(self, root_dir, target_signal='ecg'):
        self.root_dir = root_dir
        self.target_signal = target_signal
        self.file_pairs = self._get_file_pairs()

    def _get_file_pairs(self):
        file_pairs = []
        for subdir, _, files in os.walk(self.root_dir):
            l_files = [f for f in files if f.startswith('l_') and f.endswith('.csv')]
            r_files = [f for f in files if f.startswith('r_') and f.endswith('.csv')]
            ecg_files = [f for f in files if f.startswith('ecg_') and f.endswith('.csv')]
            scg_files = [f for f in files if f.startswith('scg_') and f.endswith('.csv')]

            for l_file in l_files:
                prefix = l_file.split('_')[1].split('.')[0]
                r_file = f'r_{prefix}.csv'
                ecg_file = f'ecg_{prefix}.csv'
                scg_file = f'scg_{prefix}.csv'

                if r_file in r_files and ecg_file in ecg_files and scg_file in scg_files:
                    file_pairs.append((os.path.join(subdir, l_file), os.path.join(subdir, r_file),
                                       os.path.join(subdir, ecg_file), os.path.join(subdir, scg_file)))
        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        l_file, r_file, ecg_file, scg_file = self.file_pairs[idx]

        l_data = pd.read_csv(l_file)
        r_data = pd.read_csv(r_file)
        ecg_data = pd.read_csv(ecg_file)
        scg_data = pd.read_csv(scg_file)

        l_signals = l_data[['accelerationX', 'accelerationY', 'accelerationZ', 'rotationRateX', 'rotationRateY', 'rotationRateZ', 'relative_time']].values
        r_signals = r_data[['accelerationX', 'accelerationY', 'accelerationZ', 'rotationRateX', 'rotationRateY', 'rotationRateZ', 'relative_time']].values
        scg_signals = scg_data['scg'].values.reshape(-1, 1)
        target_signals = ecg_data['ecg'].values.reshape(-1, 1) if self.target_signal == 'ecg' else scg_data['scg'].values.reshape(-1, 1)

        l_signals = (l_signals - l_signals.mean()) / l_signals.std()
        r_signals = (r_signals - r_signals.mean()) / r_signals.std()
        scg_signals = (scg_signals - scg_signals.mean()) / scg_signals.std()
        target_signals = (target_signals - target_signals.mean()) / target_signals.std()

        return {
            'left_signals': torch.tensor(l_signals, dtype=torch.float32),
            'right_signals': torch.tensor(r_signals, dtype=torch.float32),
            'scg_signals': torch.tensor(scg_signals, dtype=torch.float32),
            'target_signals': torch.tensor(target_signals, dtype=torch.float32)
        }

def collate_fn(batch, input_side='right'):
    batch_size = len(batch)
    if input_side == 'scg':
        signals = [item['scg_signals'] for item in batch]
    else:
        signals = [item[f'{input_side}_signals'] for item in batch]

    target_signals = [item['target_signals'] for item in batch]

    max_len_signals = max([len(item) for item in signals])
    max_len_target = max([len(item) for item in target_signals])

    device = signals[0].device  # 获取设备

    signals_padded = torch.zeros(batch_size, max_len_signals, signals[0].size(1), device=device)
    target_signals_padded = torch.zeros(batch_size, max_len_target, target_signals[0].size(1), device=device)

    for i in range(batch_size):
        signals_padded[i, :len(signals[i]), :] = signals[i]
        target_signals_padded[i, :len(target_signals[i]), :] = target_signals[i]

    key_padding_mask = torch.ones(batch_size, max_len_signals, device=device).bool()
    target_key_padding_mask = torch.ones(batch_size, max_len_target, device=device).bool()

    for i in range(batch_size):
        key_padding_mask[i, :len(signals[i])] = False
        target_key_padding_mask[i, :len(target_signals[i])] = False

    return {
        'signals': signals_padded,
        'target_signals': target_signals_padded,
        'target_length': torch.tensor([len(target) for target in target_signals], dtype=torch.long, device=device),
        'key_padding_mask': key_padding_mask,
        'target_key_padding_mask': target_key_padding_mask
    }
