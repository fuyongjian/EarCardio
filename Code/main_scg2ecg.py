import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from dataset_scg2ecg import BCGECDataset, collate_fn
from model import TransformerModel, LSTMModel
import GPUtil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer model for BCG to ECG/SCG prediction')
    parser.add_argument('--root_dir', type=str, default='./20SecDataset/', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in multihead attention')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--target_signal', type=str, default='ecg', choices=['ecg', 'scg'], help='Target signal to predict (ecg or scg)')
    parser.add_argument('--input_side', type=str, default='right', choices=['left', 'right', 'scg'], help='Choose input side (left, right or scg)')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['transformer', 'lstm'],
                        help='Choose model type (transformer or lstm)')
    args = parser.parse_args()
    return args

def get_free_gpu():
    GPUs = GPUtil.getGPUs()
    if not GPUs:
        return None
    freeMemory = [gpu.memoryFree for gpu in GPUs]
    free_gpu_id = freeMemory.index(max(freeMemory))
    return free_gpu_id

def save_results(output, target_signals, target_length, epoch, phase, save_dir, batch_idx, save_limit=10):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(min(save_limit, len(output))):
            output_np = output[i, :target_length[i]].cpu().numpy().flatten()
            target_np = target_signals[i, :target_length[i]].cpu().numpy().flatten()
            np.savetxt(os.path.join(save_dir, f'output_epoch_{epoch + 1}_batch_{batch_idx}_sample_{i}.csv'), output_np, delimiter=',')
            np.savetxt(os.path.join(save_dir, f'target_epoch_{epoch + 1}_batch_{batch_idx}_sample_{i}.csv'), target_np, delimiter=',')

def calculate_loss(output, target_signals, target_length, criterion):
    loss = 0
    for i in range(len(output)):
        min_length = min(output.size(1), target_length[i])
        loss += criterion(output[i, :min_length, :], target_signals[i, :min_length, :])
    loss /= len(output)
    return loss

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, input_side, num_epochs=20, patience=10):
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            signals = batch['signals'].to(device)
            target_signals = batch['target_signals'].to(device)
            target_length = batch['target_length'].to(device)
            key_padding_mask = batch['key_padding_mask'].to(device)
            target_key_padding_mask = batch['target_key_padding_mask'].to(device)

            optimizer.zero_grad()
            #output = model(signals, target_signals, src_key_padding_mask=key_padding_mask, tgt_key_padding_mask=target_key_padding_mask)
            output = model(signals, target_signals, src_key_padding_mask=key_padding_mask,
                           tgt_key_padding_mask=target_key_padding_mask, target_length=target_length)

            loss = calculate_loss(output, target_signals, target_length, criterion)

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                save_results(output, target_signals, target_length, epoch, 'train', f'results/train/epoch_{epoch + 1}', batch_idx)

        total_train_loss /= len(train_loader)
        train_losses.append(total_train_loss)

        val_loss = validate(model, val_loader, criterion, device, input_side, epoch)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
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

def validate(model, val_loader, criterion, device, input_side, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            signals = batch['signals'].to(device)
            target_signals = batch['target_signals'].to(device)
            target_length = batch['target_length'].to(device)
            key_padding_mask = batch['key_padding_mask'].to(device)
            target_key_padding_mask = batch['target_key_padding_mask'].to(device)

            output = model(signals, target_signals, src_key_padding_mask=key_padding_mask, tgt_key_padding_mask=target_key_padding_mask)

            loss = calculate_loss(output, target_signals, target_length, criterion)

            total_loss += loss.item()

            if batch_idx == 0:
                save_results(output, target_signals, target_length, epoch, 'val', f'results/val/epoch_{epoch + 1}', batch_idx)

    return total_loss / len(val_loader)

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
    input_dim = 7 if args.input_side in ['left', 'right'] else 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, input_side=args.input_side))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, input_side=args.input_side))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, input_side=args.input_side))

    #model = TransformerModel(input_dim=1 if args.input_side == 'scg' else 7, output_dim=1, d_model=args.d_model, nhead=args.nhead,
    #                         num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
    #                         dim_feedforward=args.dim_feedforward, dropout=args.dropout).to(device)

    if args.model_type == 'transformer':
        model = TransformerModel(input_dim=input_dim, output_dim=1, d_model=args.d_model, nhead=args.nhead,
                                 num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
                                 dim_feedforward=args.dim_feedforward, dropout=args.dropout).to(device)
    else:
        model = LSTMModel(input_size=input_dim, hidden_size=args.d_model, output_size=1, num_layers=args.num_encoder_layers, dropout=args.dropout).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.input_side, num_epochs=args.num_epochs, patience=args.patience)

    model.load_state_dict(torch.load('best_model.pth'))

    print("Evaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device, args.input_side, epoch=0)
    print(f'Test Loss: {test_loss:.4f}')

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            signals = batch['signals'].to(device)
            target_signals = batch['target_signals'].to(device)
            target_length = batch['target_length'].to(device)
            key_padding_mask = batch['key_padding_mask'].to(device)
            target_key_padding_mask = batch['target_key_padding_mask'].to(device)

            output = model(signals, target_signals, src_key_padding_mask=key_padding_mask, tgt_key_padding_mask=target_key_padding_mask)

            save_results(output, target_signals, target_length, epoch=0, phase='test', save_dir='results/test', batch_idx=batch_idx, save_limit=10)

if __name__ == '__main__':
    main()
