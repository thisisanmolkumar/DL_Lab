import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        x = self.dropout(self.emb(src_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed)
        return h, c


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_in_ids: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        x = self.dropout(self.emb(tgt_in_ids))
        out, (h, c) = self.lstm(x, (h, c))
        logits = self.fc_out(out)
        return logits, (h, c)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor):
        h, c = self.encoder(src, src_lens)
        logits, _ = self.decoder(tgt_in, h, c)
        return logits


@torch.no_grad()
def greedy_decode_baseline(model: Seq2Seq, src: torch.Tensor, src_lens: torch.Tensor,
                           bos_id: int, eos_id: int, max_len: int = 60) -> torch.Tensor:
    model.eval()
    h, c = model.encoder(src, src_lens)

    B = src.size(0)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)

    for _ in range(max_len):
        logits, (h, c) = model.decoder(ys[:, -1:], h, c)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if torch.all(next_tok.squeeze(1) == eos_id):
            break
    return ys


class EncoderLSTMWithOutputs(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        x = self.dropout(self.emb(src_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src_ids.size(1)
        )
        return enc_out, (h, c)


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W_h = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out: torch.Tensor, dec_h_t: torch.Tensor, src_mask: torch.Tensor):
        energy = torch.tanh(self.W_h(enc_out) + self.W_s(dec_h_t).unsqueeze(1))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn_w = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_w.unsqueeze(1), enc_out).squeeze(1)
        return context, attn_w


class AttnDecoderBahdanau(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 attn: BahdanauAttention, num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.attn = attn
        self.lstm = nn.LSTM(emb_dim + hidden_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim, vocab_size)

    def forward_step(self, prev_tok: torch.Tensor, state, enc_out: torch.Tensor, src_mask: torch.Tensor):
        h, c = state
        dec_h_t = h[-1]
        context, attn_w = self.attn(enc_out, dec_h_t, src_mask)
        emb = self.dropout(self.emb(prev_tok))
        lstm_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, (h, c) = self.lstm(lstm_in, (h, c))
        out_cat = torch.cat([out.squeeze(1), context], dim=-1)
        logits = self.fc_out(out_cat)
        return logits, (h, c), attn_w

    def forward(self, tgt_in: torch.Tensor, init_state, enc_out: torch.Tensor, src_mask: torch.Tensor):
        B, Ttgt = tgt_in.shape
        logits_all, attn_all = [], []
        state = init_state
        for t in range(Ttgt):
            prev_tok = tgt_in[:, t:t+1]
            logits, state, attn_w = self.forward_step(
                prev_tok, state, enc_out, src_mask)
            logits_all.append(logits.unsqueeze(1))
            attn_all.append(attn_w.unsqueeze(1))
        return torch.cat(logits_all, dim=1), torch.cat(attn_all, dim=1)


class Seq2SeqBahdanau(nn.Module):
    def __init__(self, encoder: EncoderLSTMWithOutputs, decoder: AttnDecoderBahdanau, pad_id_src: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id_src = pad_id_src

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_id_src)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor):
        enc_out, (h, c) = self.encoder(src, src_lens)
        src_mask = self.make_src_mask(src)
        logits, attn = self.decoder(tgt_in, (h, c), enc_out, src_mask)
        return logits, attn


@torch.no_grad()
def greedy_decode_bahdanau(model: Seq2SeqBahdanau, src: torch.Tensor, src_lens: torch.Tensor,
                           bos_id: int, eos_id: int, max_len: int = 60, return_attn: bool = True):
    model.eval()
    enc_out, (h, c) = model.encoder(src, src_lens)
    src_mask = model.make_src_mask(src)

    B = src.size(0)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)

    attn_steps = []
    state = (h, c)

    for _ in range(max_len):
        prev_tok = ys[:, -1:]
        logits, state, attn_w = model.decoder.forward_step(
            prev_tok, state, enc_out, src_mask)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if return_attn:
            attn_steps.append(attn_w.unsqueeze(1))
        if torch.all(next_tok.squeeze(1) == eos_id):
            break

    attn_mat = torch.cat(attn_steps, dim=1) if (
        return_attn and attn_steps) else None
    return ys, attn_mat


class LuongAttention(nn.Module):
    def __init__(self, hidden_dim: int, method: str = "general"):
        super().__init__()
        assert method in {"dot", "general"}
        self.method = method
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim,
                           bias=False) if method == "general" else None

    def score(self, dec_h_t: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        if self.method == "dot":
            return torch.bmm(enc_out, dec_h_t.unsqueeze(-1)).squeeze(-1)
        else:
            enc_proj = self.W(enc_out)
            return torch.bmm(enc_proj, dec_h_t.unsqueeze(-1)).squeeze(-1)

    def forward(self, enc_out: torch.Tensor, dec_h_t: torch.Tensor, src_mask: torch.Tensor):
        scores = self.score(dec_h_t, enc_out)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn_w = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_w.unsqueeze(1), enc_out).squeeze(1)
        return context, attn_w


class AttnDecoderLuong(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 attn: LuongAttention, num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.attn = attn

        self.lstm = nn.LSTM(emb_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc_comb = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, prev_tok: torch.Tensor, state, enc_out: torch.Tensor, src_mask: torch.Tensor):
        h, c = state
        emb = self.dropout(self.emb(prev_tok))
        dec_out, (h, c) = self.lstm(emb, (h, c))
        dec_h_t = dec_out.squeeze(1)

        context, attn_w = self.attn(enc_out, dec_h_t, src_mask)
        attn_hidden = torch.tanh(self.fc_comb(
            torch.cat([dec_h_t, context], dim=-1)))
        logits = self.fc_out(attn_hidden)

        return logits, (h, c), attn_w

    def forward(self, tgt_in: torch.Tensor, init_state, enc_out: torch.Tensor, src_mask: torch.Tensor):
        B, Ttgt = tgt_in.shape
        logits_all = []
        attn_all = []

        state = init_state
        for t in range(Ttgt):
            prev_tok = tgt_in[:, t:t+1]
            logits, state, attn_w = self.forward_step(
                prev_tok, state, enc_out, src_mask)
            logits_all.append(logits.unsqueeze(1))
            attn_all.append(attn_w.unsqueeze(1))

        return torch.cat(logits_all, dim=1), torch.cat(attn_all, dim=1)


class Seq2SeqLuong(nn.Module):
    def __init__(self, encoder: EncoderLSTMWithOutputs, decoder: AttnDecoderLuong, pad_id_src: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id_src = pad_id_src

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_id_src)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor):
        enc_out, (h, c) = self.encoder(src, src_lens)
        src_mask = self.make_src_mask(src)
        logits, attn = self.decoder(tgt_in, (h, c), enc_out, src_mask)
        return logits, attn


@torch.no_grad()
def greedy_decode_luong(model: Seq2SeqLuong, src: torch.Tensor, src_lens: torch.Tensor,
                        bos_id: int, eos_id: int, max_len: int = 60, return_attn: bool = True):
    model.eval()
    enc_out, (h, c) = model.encoder(src, src_lens)
    src_mask = model.make_src_mask(src)

    B = src.size(0)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)

    attn_steps = []
    state = (h, c)

    for _ in range(max_len):
        prev_tok = ys[:, -1:]
        logits, state, attn_w = model.decoder.forward_step(
            prev_tok, state, enc_out, src_mask)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)

        if return_attn:
            attn_steps.append(attn_w.unsqueeze(1))

        if torch.all(next_tok.squeeze(1) == eos_id):
            break

    if return_attn:
        attn_mat = torch.cat(attn_steps, dim=1) if attn_steps else torch.empty(
            (B, 0, enc_out.size(1)), device=src.device)
        return ys, attn_mat
    return ys, None
