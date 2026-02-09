import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset import BCGECDataset, collate_fn
from model import TransformerModel, LSTMModel
import GPUtil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Transformer model for BCG to ECG/SCG prediction')
    parser.add_argument('--root_dir', type=str, default='./Dataset/testDataset/M1/', help='Dataset directory')
    parser.add_argument('--results_dir', type=str, default='./results/test/', help='Directory to save results and logs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in multihead attention')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--target_signal', type=str, default='ecg', choices=['ecg', 'scg'], help='Target signal to predict (ecg or scg)')
    parser.add_argument('--input_side', type=str, default='right', choices=['left', 'right', 'dual', 'scg'], help='Choose input side (left, right, dual or scg)')
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'lstm'], help='Choose model type (transformer or lstm)')
    parser.add_argument('--pos_encoding', type=str, default='cosine', choices=['timestamp', 'cosine', 'cosine_no_time'], help='Choose positional encoding method (timestamp or cosine)')
    parser.add_argument('--model_path', type=str, default='/results/input_length/0Sec/', help='Path to the saved model')
    args = parser.parse_args()
    return args

def get_free_gpu():
    GPUs = GPUtil.getGPUs()
    if not GPUs:
        return None
    freeMemory = [gpu.memoryFree for gpu in GPUs]
    free_gpu_id = freeMemory.index(max(freeMemory))
    return free_gpu_id

def save_all_results(output, target_signals, target_length, phase, save_dir, batch_idx):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(len(output)):
            output_np = output[i, :target_length[i]].cpu().numpy().flatten()
            target_np = target_signals[i, :target_length[i]].cpu().numpy().flatten()
            np.savetxt(os.path.join(save_dir, f'{phase}_batch_{batch_idx}_sample_{i}_output.csv'), output_np, delimiter=',')
            np.savetxt(os.path.join(save_dir, f'{phase}_batch_{batch_idx}_sample_{i}_target.csv'), target_np, delimiter=',')

def calculate_loss(output, target_signals, target_length, criterion):
    loss = 0
    for i in range(len(output)):
        min_length = min(output.size(1), target_length[i])
        loss += criterion(output[i, :min_length, :], target_signals[i, :min_length, :])
    loss /= len(output)
    return loss

def run_test(model, test_loader, criterion, device, input_side, results_dir):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            signals = batch['signals'].to(device)
            target_signals = batch['target_signals'].to(device)
            target_length = batch['target_length'].to(device)
            key_padding_mask = batch['key_padding_mask'].to(device)
            target_key_padding_mask = batch['target_key_padding_mask'].to(device)

            output = model(signals, target_signals, src_key_padding_mask=key_padding_mask, tgt_key_padding_mask=target_key_padding_mask)

            loss = calculate_loss(output, target_signals, target_length, criterion)
            total_loss += loss.item()

            save_all_results(output, target_signals, target_length, phase='test', save_dir=os.path.join(results_dir, 'test'), batch_idx=batch_idx)

    return total_loss / len(test_loader)

def main():
    args = parse_args()

    free_gpu_id = get_free_gpu()
    device = torch.device(f"cuda:{free_gpu_id}" if free_gpu_id is not None else "cpu")
    print(f"Using device: {device}")

    dataset = BCGECDataset(args.root_dir, target_signal=args.target_signal)
    test_size = len(dataset)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, input_side=args.input_side))

    input_dim = 6 if args.pos_encoding == 'cosine_no_time' and args.input_side != 'scg' else (
        6 if args.pos_encoding == 'timestamp' and args.input_side in ['left', 'right', 'dual'] else 7 if args.input_side in ['left', 'right', 'dual'] else 1)

    if args.model_type == 'transformer':
        model = TransformerModel(input_dim=input_dim, output_dim=1, d_model=args.d_model, nhead=args.nhead,
                                 num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
                                 dim_feedforward=args.dim_feedforward, dropout=args.dropout, pos_encoding=args.pos_encoding).to(device)
    else:
        model = LSTMModel(input_size=input_dim, hidden_size=args.d_model, output_size=1, num_layers=args.num_encoder_layers, dropout=args.dropout).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    criterion = nn.L1Loss()

    print("Evaluating on test set...")
    test_loss = run_test(model, test_loader, criterion, device, args.input_side, results_dir=args.results_dir)
    print(f'Test Loss: {test_loss:.4f}')

    with open(os.path.join(args.results_dir, 'test_log.log'), 'a') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')

if __name__ == '__main__':
    main()
