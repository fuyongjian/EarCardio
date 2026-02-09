import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from dataset import BCGECDataset, collate_fn
from model import TransformerModel, LSTMModel
import GPUtil
import argparse
import random
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer model for BCG to ECG/SCG prediction')
    parser.add_argument('--root_dir', type=str, default='./Dataset/5SecDataset/', help='Dataset directory')
    parser.add_argument('--results_dir', type=str, default='./results/test/', help='Directory to save results and logs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in multihead attention')
    parser.add_argument('--num_encoder_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=128, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=5, help='Learning rate scheduler patience')
    parser.add_argument('--target_signal', type=str, default='ecg', choices=['ecg', 'scg'], help='Target signal to predict (ecg or scg)')
    parser.add_argument('--input_side', type=str, default='right', choices=['left', 'right', 'dual', 'scg'], help='Choose input side (left, right, dual or scg)')
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'lstm'], help='Choose model type (transformer or lstm)')
    parser.add_argument('--pos_encoding', type=str, default='cosine', choices=['timestamp', 'cosine', 'cosine_no_time'], help = 'Choose positional encoding method (timestamp or cosine)')
    parser.add_argument('--seed', type=int, default=999, help='Random seed for dataset splitting, 0 means random')
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
def inference(model, src, src_key_padding_mask=None, max_len=500):
    model.eval()
    memory = model.transformer_encoder(model.encoder_input_linear(src) * math.sqrt(model.d_model), src_key_padding_mask=src_key_padding_mask)
    tgt = torch.zeros(src.size(0), 1, 1).to(src.device)  # 初始化第一个解码器输入
    # 初始化解码器的第一个输入

    for i in range(max_len):
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        linear_tgt = model.decoder_input_linear(tgt) * torch.sqrt(torch.tensor(model.d_model, dtype=torch.float32)).to(
            src.device)
        encoded_tgt = model.pos_encoder(linear_tgt)
        out = model.transformer_decoder(encoded_tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        out = model.output_linear(out[:, -1, :]).unsqueeze(1)
        tgt = torch.cat([tgt, out], dim=1)  # 将当前步的预测结果作为下一个时间步的输入

    return tgt

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, input_side, num_epochs=20, patience=10, results_dir='./results/'):
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    log_file = os.path.join(results_dir, 'training_log.log')
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        model.teacher_forcing_ratio = max(0.5, 1.0 - epoch / num_epochs)
        for batch_idx, batch in enumerate(train_loader):
            signals = batch['signals'].to(device)
            target_signals = batch['target_signals'].to(device)
            target_length = batch['target_length'].to(device)
            key_padding_mask = batch['key_padding_mask'].to(device)
            target_key_padding_mask = batch['target_key_padding_mask'].to(device)

            optimizer.zero_grad()
            output = model(signals, target_signals, src_key_padding_mask=key_padding_mask, tgt_key_padding_mask=target_key_padding_mask, target_length=target_length)
            #output = inference(model, signals, src_key_padding_mask=key_padding_mask, max_len=target_signals.size(1))
            loss = calculate_loss(output, target_signals, target_length, criterion)

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                save_results(output, target_signals, target_length, epoch, 'train', os.path.join(results_dir, f'train/epoch_{epoch + 1}'), batch_idx)

        total_train_loss /= len(train_loader)
        train_losses.append(total_train_loss)

        val_loss = validate(model, val_loader, criterion, device, input_side, epoch, results_dir)
        val_losses.append(val_loss)

        with open(log_file, 'a') as f:
            f.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Validation Loss: {val_loss:.4f}\n')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
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
    plt.savefig(os.path.join(results_dir, 'loss_plot.png'))
    plt.show()

def validate(model, val_loader, criterion, device, input_side, epoch, results_dir):
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
            #output = inference(model, signals, src_key_padding_mask=key_padding_mask, max_len=target_signals.size(1))
            loss = calculate_loss(output, target_signals, target_length, criterion)

            total_loss += loss.item()

            if batch_idx == 0:
                save_results(output, target_signals, target_length, epoch, 'val', os.path.join(results_dir, f'val/epoch_{epoch + 1}'), batch_idx)

    return total_loss / len(val_loader)

def main():
    args = parse_args()

    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    free_gpu_id = get_free_gpu()
    device = torch.device(f"cuda:{free_gpu_id}" if free_gpu_id is not None else "cpu")
    print(f"Using device: {device}")

    dataset = BCGECDataset(args.root_dir, target_signal=args.target_signal)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    input_dim = 6 if args.pos_encoding == 'cosine_no_time' and args.input_side != 'scg' else (
        6 if args.pos_encoding == 'timestamp' and args.input_side in ['left', 'right', 'dual'] else 7 if args.input_side in ['left', 'right', 'dual'] else 1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, input_side=args.input_side))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, input_side=args.input_side))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, input_side=args.input_side))

    if args.model_type == 'transformer':
        model = TransformerModel(input_dim=input_dim, output_dim=1, d_model=args.d_model, nhead=args.nhead,
                                 num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
                                 dim_feedforward=args.dim_feedforward, dropout=args.dropout, pos_encoding=args.pos_encoding).to(device)
    else:
        model = LSTMModel(input_size=input_dim, hidden_size=args.d_model, output_size=1, num_layers=args.num_encoder_layers, dropout=args.dropout).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, factor=0.5)

    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.input_side, num_epochs=args.num_epochs, patience=args.patience, results_dir=args.results_dir)

    model.load_state_dict(torch.load(os.path.join(args.results_dir, 'best_model.pth')))

    print("Evaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device, args.input_side, epoch=0, results_dir=args.results_dir)
    print(f'Test Loss: {test_loss:.4f}')

    with open(os.path.join(args.results_dir, 'training_log.log'), 'a') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            signals = batch['signals'].to(device)
            target_signals = batch['target_signals'].to(device)
            target_length = batch['target_length'].to(device)
            key_padding_mask = batch['key_padding_mask'].to(device)
            target_key_padding_mask = batch['target_key_padding_mask'].to(device)
            output = model(signals, target_signals, src_key_padding_mask=key_padding_mask,
                           tgt_key_padding_mask=target_key_padding_mask)
            #output = inference(model, signals, src_key_padding_mask=key_padding_mask, max_len=target_signals.size(1))
            #output = inference(signals, target_signals, src_key_padding_mask=key_padding_mask, tgt_key_padding_mask=target_key_padding_mask)

            save_all_results(output, target_signals, target_length, phase='test', save_dir=os.path.join(args.results_dir, 'test'), batch_idx=batch_idx)

if __name__ == '__main__':
    main()
