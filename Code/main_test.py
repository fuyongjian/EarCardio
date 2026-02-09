import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset import BCGECDataset, collate_fn
from transformer import TransformerModel
import argparse
import GPUtil
import numpy as np

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer model for BCG to ECG/SCG prediction')
    parser.add_argument('--root_dir', type=str, default='./Dataset/', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in multihead attention')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--positional_encoding', type=str, default='cosine', choices=['time', 'cosine'],
                        help='Type of positional encoding')
    parser.add_argument('--target_signal', type=str, default='ecg', choices=['ecg', 'scg'],
                        help='Target signal to predict (ecg or scg)')
    args = parser.parse_args()
    return args

def get_free_gpu():
    GPUs = GPUtil.getGPUs()
    if not GPUs:
        return None
    freeMemory = [gpu.memoryFree for gpu in GPUs]
    free_gpu_id = freeMemory.index(max(freeMemory))
    return free_gpu_id

# Train function with validation
def train(model, train_loader, val_loader, criterion_smooth_l1, criterion_l2, optimizer, scheduler, device,
          num_epochs=20, patience=10):
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_signals = batch['r_signals'].to(device)
            target_signals = batch['target_signals'].to(device)
            input_time = batch['r_time'].to(device)
            target_length = batch['target_length'].to(device)

            tgt = torch.zeros((input_signals.size(0), input_signals.size(1), input_signals.size(2))).to(input_signals.device)
            tgt_timestamps = torch.arange(0, tgt.size(1)).unsqueeze(0).repeat(tgt.size(0), 1).float().to(input_signals.device)

            optimizer.zero_grad()
            output = model(input_signals, tgt, input_time, tgt_timestamps)

            loss_smooth_l1 = 0
            loss_l2 = 0
            for i in range(len(output)):
                min_length = min(output.size(1), target_length[i])
                loss_smooth_l1 += criterion_smooth_l1(output[i, :min_length, :], target_signals[i, :min_length, :])
                loss_l2 += criterion_l2(output[i, :min_length, :], target_signals[i, :min_length, :])
            loss_smooth_l1 /= len(output)
            loss_l2 /= len(output)

            total_train_loss += loss_smooth_l1.item()  # Use Smooth L1 loss for training

            loss_smooth_l1.backward()
            optimizer.step()

        total_train_loss /= len(train_loader)
        train_losses.append(total_train_loss)

        val_loss_smooth_l1, val_loss_l2 = validate(model, val_loader, criterion_smooth_l1, criterion_l2, device, epoch)
        val_losses.append(val_loss_smooth_l1)  # Use Smooth L1 loss for validation

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Smooth L1 Loss: {total_train_loss:.4f}, Validation Smooth L1 Loss: {val_loss_smooth_l1:.4f}, Validation L2 Loss: {val_loss_l2:.4f}')

        scheduler.step(val_loss_smooth_l1)

        if val_loss_smooth_l1 < best_loss:
            best_loss = val_loss_smooth_l1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

def save_predictions(output, target_signals, target_lengths, epoch, phase, batch_idx, save_limit=20):
    save_dir = f'predictions/{phase}_epoch_{epoch + 1}'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(min(save_limit, len(output))):
            output_np = output[i, :target_lengths[i]].cpu().numpy().flatten()
            target_np = target_signals[i, :target_lengths[i]].cpu().numpy().flatten()
            np.savetxt(os.path.join(save_dir, f'output_batch_{batch_idx}_sample_{i}.csv'), output_np, delimiter=',')
            np.savetxt(os.path.join(save_dir, f'target_batch_{batch_idx}_sample_{i}.csv'), target_np, delimiter=',')

def validate(model, val_loader, criterion_smooth_l1, criterion_l2, device, epoch):
    model.eval()
    total_loss_smooth_l1 = 0
    total_loss_l2 = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_signals = batch['r_signals'].to(device)
            target_signals = batch['target_signals'].to(device)
            input_time = batch['r_time'].to(device)
            target_lengths = batch['target_length'].to(device)

            tgt = torch.zeros((input_signals.size(0), input_signals.size(1), input_signals.size(2))).to(input_signals.device)
            tgt_timestamps = torch.arange(0, tgt.size(1)).unsqueeze(0).repeat(tgt.size(0), 1).float().to(input_signals.device)

            output = model(input_signals, tgt, input_time, tgt_timestamps)

            loss_smooth_l1 = 0
            loss_l2 = 0
            for i in range(len(output)):
                min_length = min(output.size(1), target_lengths[i])
                loss_smooth_l1 += criterion_smooth_l1(output[i, :min_length, :], target_signals[i, :min_length, :])
                loss_l2 += criterion_l2(output[i, :min_length, :], target_signals[i, :min_length, :])
            loss_smooth_l1 /= len(output)
            loss_l2 /= len(output)

            total_loss_smooth_l1 += loss_smooth_l1.item()
            total_loss_l2 += loss_l2.item()

            if batch_idx < 1:
                save_predictions(output, target_signals, target_lengths, epoch, 'val', batch_idx, save_limit=20)

    return total_loss_smooth_l1 / len(val_loader), total_loss_l2 / len(val_loader)

def main():
    args = parse_args()

    free_gpu_id = get_free_gpu()
    device = torch.device(f"cuda:{free_gpu_id}" if free_gpu_id is not None else "cpu")
    print(f"Using device: {device}")

    dataset = BCGECDataset(args.root_dir, target_signal=args.target_signal)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerModel(input_dim=6, output_dim=1, d_model=args.d_model, nhead=args.nhead,
                             num_encoder_layers=args.num_encoder_layers,
                             num_decoder_layers=args.num_decoder_layers, dim_feedforward=args.dim_feedforward,
                             dropout=args.dropout,
                             positional_encoding=args.positional_encoding).to(device)

    criterion_smooth_l1 = nn.SmoothL1Loss()
    criterion_l2 = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # Add weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print("Starting training...")
    train(model, train_loader, val_loader, criterion_smooth_l1, criterion_l2, optimizer, scheduler, device,
          num_epochs=args.num_epochs, patience=args.patience)

    model.load_state_dict(torch.load('best_model.pth'))

    print("Evaluating on test set...")
    test_loss_smooth_l1, test_loss_l2 = validate(model, test_loader, criterion_smooth_l1, criterion_l2, device, epoch=0)
    print(f'Test Smooth L1 Loss: {test_loss_smooth_l1:.4f}, Test L2 Loss: {test_loss_l2:.4f}')

if __name__ == '__main__':
    main()
