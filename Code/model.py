import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x, tgt=None, src_key_padding_mask=None, tgt_key_padding_mask=None, target_length=None):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        if target_length is not None:
            pool_size = target_length.max().item()
            x = F.adaptive_avg_pool1d(x, pool_size)  # 动态调整序列长度到目标长度
        x = x.transpose(1, 2)
        h_lstm, _ = self.lstm(x)
        x = self.relu(self.fc1(h_lstm))
        output = self.fc2(x)
        return output


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=d_model ** -0.5)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=5000, pos_encoding='timestamp'):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pos_encoding = pos_encoding
        self.encoder_input_linear = nn.Linear(input_dim, d_model)
        self.decoder_input_linear = nn.Linear(output_dim, d_model)
        self.output_linear = nn.Linear(d_model, output_dim)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        self.teacher_forcing_ratio = 1.0  # 初始化teacher forcing比率

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, target_length=None):
        # 编码器输入线性变换
        if self.pos_encoding == 'cosine':
            src = self.encoder_input_linear(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
        elif self.pos_encoding == 'cosine_no_time':
            src = src[:, :, :-1]
            timestamps = src[:, :, -1]
            src = self.encoder_input_linear(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
        else:
            timestamps = src[:, :, -1]
            src = src[:, :, :-1]
            src = self.encoder_input_linear(src) * math.sqrt(self.d_model)
            src = src + timestamps.unsqueeze(-1)

        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # 解码器输入线性变换
        batch_size = tgt.size(0)
        seq_len = tgt.size(1)
        decoder_input = torch.zeros(batch_size, seq_len, self.d_model, device=tgt.device)
        output = torch.zeros(batch_size, seq_len, tgt.size(2), device=tgt.device)

        if self.training:  # 训练时使用Scheduled Sampling
            for t in range(seq_len):
                if t == 0 or torch.rand(1).item() < self.teacher_forcing_ratio:
                    decoder_input_step = self.decoder_input_linear(tgt[:, t, :]) * math.sqrt(self.d_model)
                else:
                    decoder_input_step = self.decoder_input_linear(output[:, t - 1, :].detach()) * math.sqrt(self.d_model)
                decoder_input_step = self.pos_encoder(decoder_input_step.unsqueeze(1)).squeeze(1)
                decoder_input = decoder_input.clone()
                decoder_input[:, t, :] = decoder_input_step  # 避免就地操作

                # 使用所有先前时间步的输入进行解码
                decoder_output = self.transformer_decoder(decoder_input[:, :t + 1, :], memory,
                                                          memory_key_padding_mask=src_key_padding_mask)
                output = output.clone()
                output[:, t, :] = self.output_linear(decoder_output[:, -1, :])

        else:  # 测试时仅使用模型预测的结果
            for t in range(seq_len):
                if t == 0:
                    decoder_input_step = self.decoder_input_linear(tgt[:, t, :]) * math.sqrt(self.d_model)
                else:
                    decoder_input_step = self.decoder_input_linear(output[:, t - 1, :]) * math.sqrt(self.d_model)
                decoder_input_step = self.pos_encoder(decoder_input_step.unsqueeze(1)).squeeze(1)
                decoder_input = decoder_input.clone()
                decoder_input[:, t, :] = decoder_input_step  # 避免就地操作

                # 使用所有先前时间步的输入进行解码
                decoder_output = self.transformer_decoder(decoder_input[:, :t + 1, :], memory,
                                                          memory_key_padding_mask=src_key_padding_mask)
                output = output.clone()
                output[:, t, :] = self.output_linear(decoder_output[:, -1, :])

        return output
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).to(device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def pad_sequence(sequences, batch_first=False, padding_value=0.0):
        max_size = sequences[0].size()
        trailing_dims = max_size[1:]
        max_len = max([s.size(0) for s in sequences])
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims

        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor

        return out_tensor


if __name__ == "__main__":
    input_dim = 6  # BCG信号的维度
    time_dim = 1  # 时间信息的维度
    d_model = 512  # Transformer模型的维度
    output_dim = 1  # ECG信号的维度

    model = TransformerModel(input_dim + time_dim, output_dim, d_model, nhead=8, num_encoder_layers=6,
                             dim_feedforward=2048, dropout=0.1)

    # 生成具有可变长度的合成BCG信号数据
    batch_size = 32
    bcg_seq_lens = [torch.randint(115, 126, (1,)).item() for _ in range(batch_size)]
    ecg_seq_lens = [torch.randint(65, 131, (1,)).item() for _ in range(batch_size)]  # ECG信号长度

    bcg_signals = [torch.randn(seq_len, input_dim) for seq_len in bcg_seq_lens]
    ecg_signals = [torch.randn(seq_len, output_dim) for seq_len in ecg_seq_lens]

    # 生成相对时间信息
    bcg_time_stamps = [torch.linspace(0, 1, steps=seq_len).unsqueeze(1) for seq_len in bcg_seq_lens]
    # 将时间信息与BCG信号结合
    bcg_with_time = [torch.cat((bcg, time), dim=1) for bcg, time in zip(bcg_signals, bcg_time_stamps)]

    # 填充序列到相同长度
    bcg_signals_padded = TransformerModel.pad_sequence(bcg_with_time, batch_first=True)
    ecg_signals_padded = TransformerModel.pad_sequence(ecg_signals, batch_first=True)

    # 创建填充掩码
    bcg_key_padding_mask = torch.zeros(bcg_signals_padded.size(0), bcg_signals_padded.size(1)).bool()
    ecg_key_padding_mask = torch.zeros(ecg_signals_padded.size(0), ecg_signals_padded.size(1)).bool()

    for i, seq_len in enumerate(bcg_seq_lens):
        if seq_len < bcg_signals_padded.size(1):
            bcg_key_padding_mask[i, seq_len:] = True

    for i, seq_len in enumerate(ecg_seq_lens):
        if seq_len < ecg_signals_padded.size(1):
            ecg_key_padding_mask[i, seq_len:] = True

    # 前向传播
    ecg_predictions_padded = model(bcg_signals_padded, ecg_signals_padded, src_key_padding_mask=bcg_key_padding_mask,
                                   tgt_key_padding_mask=ecg_key_padding_mask)

    # 从填充的预测结果中提取实际长度的预测结果
    ecg_predictions = [ecg_predictions_padded[i, :seq_len] for i, seq_len in enumerate(ecg_seq_lens)]

    # 打印形状以检查
    print("BCG signals padded shape:", bcg_signals_padded.shape)
    print("ECG signals padded shape:", ecg_signals_padded.shape)
    for i, (pred, actual_len) in enumerate(zip(ecg_predictions, ecg_seq_lens)):
        print(f"ECG prediction {i} shape: {pred.shape}, expected length: {actual_len}")

    # 检查输出形状是否符合预期
    for i, (pred, actual_len) in enumerate(zip(ecg_predictions, ecg_seq_lens)):
        assert pred.shape[0] == actual_len, f"Output shape mismatch at index {i}!"
    print("Model output is correct!")
