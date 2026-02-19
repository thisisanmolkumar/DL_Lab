import random
import re
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import models

DEVICE = "mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu"

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
DATA_PATH = Path("spa.txt")

SAMPLE_N = 10_000

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

MIN_FREQ = 2
MAX_VOCAB_SRC = 20_000
MAX_VOCAB_TGT = 20_000

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def normalize(text: str, lowercase: bool = True) -> str:
    text = text.strip()
    if lowercase:
        text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str):
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def encode_src(sentence: str, src_vocab, lowercase=True, max_len=50):
    toks = tokenize(normalize(sentence, lowercase))
    toks = toks[:max_len]
    ids = [src_vocab["stoi"].get(t, src_vocab["stoi"][UNK]) for t in toks]
    ids.append(src_vocab["stoi"][EOS])
    return ids


def ids_to_sentence(ids, itos):
    toks = []
    for i in ids:
        tok = itos[int(i)]
        if tok in (PAD, BOS):
            continue
        if tok == EOS:
            break
        toks.append(tok)
    sent = " ".join(toks)
    sent = re.sub(r"\s+([.,!?;:])", r"\1", sent)
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent


def build_vocab_from_pairs(pairs, side: str, min_freq: int = 2, max_vocab: Optional[int] = None) -> Dict[str, Any]:
    assert side in {"src", "tgt"}
    idx = 0 if side == "src" else 1

    counter = Counter()
    for en, es in pairs:
        text = normalize(en if idx == 0 else es, True)
        toks = tokenize(text)
        counter.update(toks)

    itos = list(SPECIAL_TOKENS)
    for w, c in counter.most_common():
        if c < min_freq:
            break
        if max_vocab is not None and len(itos) >= max_vocab:
            break
        if w not in SPECIAL_TOKENS:
            itos.append(w)

    stoi = {w: i for i, w in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}


def load_baseline_best(ckpt_path: str, src_vocab, tgt_vocab):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    enc = models.EncoderLSTM(len(src_vocab["itos"]), cfg["EMB_SRC"], cfg["HIDDEN"],
                             cfg["NUM_LAYERS"], cfg["DROPOUT"], src_vocab["stoi"][PAD])
    dec = models.DecoderLSTM(len(tgt_vocab["itos"]), cfg["EMB_TGT"], cfg["HIDDEN"],
                             cfg["NUM_LAYERS"], cfg["DROPOUT"], tgt_vocab["stoi"][PAD])
    m = models.Seq2Seq(enc, dec).to(DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


def load_bahdanau_best(ckpt_path: str, src_vocab, tgt_vocab):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    enc = models.EncoderLSTMWithOutputs(len(src_vocab["itos"]), cfg["EMB_SRC"],
                                        cfg["HIDDEN"], cfg["NUM_LAYERS"], cfg["DROPOUT"], src_vocab["stoi"][PAD])
    attn = models.BahdanauAttention(cfg["HIDDEN"], cfg["HIDDEN"], 256)
    dec = models.AttnDecoderBahdanau(len(tgt_vocab["itos"]), cfg["EMB_TGT"], cfg["HIDDEN"],
                                     attn, cfg["NUM_LAYERS"], cfg["DROPOUT"], tgt_vocab["stoi"][PAD])
    m = models.Seq2SeqBahdanau(
        enc, dec, pad_id_src=src_vocab["stoi"][PAD]).to(DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


def load_luong_best(ckpt_path: str, src_vocab, tgt_vocab, method="general"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    enc = models.EncoderLSTMWithOutputs(len(src_vocab["itos"]), cfg["EMB_SRC"],
                                        cfg["HIDDEN"], cfg["NUM_LAYERS"], cfg["DROPOUT"], src_vocab["stoi"][PAD])
    attn = models.LuongAttention(cfg["HIDDEN"], method=method)
    dec = models.AttnDecoderLuong(len(tgt_vocab["itos"]), cfg["EMB_TGT"], cfg["HIDDEN"],
                                  attn, cfg["NUM_LAYERS"], cfg["DROPOUT"], tgt_vocab["stoi"][PAD])
    m = models.Seq2SeqLuong(
        enc, dec, pad_id_src=src_vocab["stoi"][PAD]).to(DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@torch.no_grad()
def translate_with(model_type: str, model, sentence: str, src_vocab, tgt_vocab, lowercase=True):
    src_ids = encode_src(sentence, src_vocab, lowercase=lowercase)
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long, device=DEVICE)

    bos_id = tgt_vocab["stoi"][BOS]
    eos_id = tgt_vocab["stoi"][EOS]

    if model_type == "baseline":
        pred = models.greedy_decode_baseline(model, src, src_lens, bos_id, eos_id, max_len=60)[
            0].cpu().tolist()
    elif model_type == "bahdanau":
        pred = models.greedy_decode_bahdanau(model, src, src_lens, bos_id, eos_id, max_len=60)[
            0].cpu().tolist()[0]
    elif model_type == "luong":
        pred = models.greedy_decode_luong(model, src, src_lens, bos_id, eos_id, max_len=60)[
            0].cpu().tolist()[0]
    else:
        raise ValueError("model_type must be one of: baseline/bahdanau/luong")

    return ids_to_sentence(pred, tgt_vocab["itos"])


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            en, es = parts[0], parts[1]
            pairs.append((en, es))
    return pairs


def split_pairs(
    pairs: List[Tuple[str, str]],
    sample_n: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)

    if sample_n is not None:
        pairs = pairs[:sample_n]

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    return train, val, test


def predict(text: str):
    pairs = load_pairs(DATA_PATH)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        sample_n=SAMPLE_N,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    src_vocab = build_vocab_from_pairs(
        train_pairs, "src", min_freq=MIN_FREQ, max_vocab=MAX_VOCAB_SRC)
    tgt_vocab = build_vocab_from_pairs(
        train_pairs, "tgt", min_freq=MIN_FREQ, max_vocab=MAX_VOCAB_TGT)

    baseline = load_baseline_best(
        "models/baseline_no_attention_best.pt", src_vocab, tgt_vocab)
    bahdanau = load_bahdanau_best(
        "models/bahdanau_best.pt", src_vocab, tgt_vocab)
    luong = load_luong_best("models/luong_best.pt",
                            src_vocab, tgt_vocab, method="general")

    print("EN:", text)
    print("Baseline :", translate_with(
        "baseline", baseline, text, src_vocab, tgt_vocab))
    print("Bahdanau :", translate_with(
        "bahdanau", bahdanau, text, src_vocab, tgt_vocab))
    print("Luong    :", translate_with(
        "luong", luong, text, src_vocab, tgt_vocab))

    return {
        "EN": text,
        "Baseline": translate_with("baseline", baseline, text, src_vocab, tgt_vocab),
        "Bahdanau": translate_with("bahdanau", bahdanau, text, src_vocab, tgt_vocab),
        "Luong": translate_with("luong", luong, text, src_vocab, tgt_vocab)
    }
