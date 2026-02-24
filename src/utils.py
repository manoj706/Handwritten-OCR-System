import torch
import numpy as np
from typing import List

class LabelEncoder:
    def __init__(self, characters: str):
        self.characters = sorted(list(set(characters)))
        self.char_to_int = {c: i + 1 for i, c in enumerate(self.characters)} # 0 is reserved for CTC blank
        self.int_to_char = {i + 1: c for i, c in enumerate(self.characters)}
        self.blank_idx = 0

    def encode(self, text: str) -> List[int]:
        return [self.char_to_int[c] for c in text if c in self.char_to_int]

    def decode(self, indices: List[int], remove_blank: bool = True) -> str:
        res = []
        for i in indices:
            if remove_blank and i == self.blank_idx:
                continue
            if i in self.int_to_char:
                res.append(self.int_to_char[i])
        return "".join(res)

    def decode_greedy(self, logits: torch.Tensor) -> List[str]:
        """
        Decodes a batch of logits using greedy decoding.
        logits: (N, T, C) - Expects batch first.
        """
        if logits.ndim == 3:
            # Argmax
            preds = torch.argmax(logits, dim=2).detach().cpu().numpy() # (N, T)
        else:
             # Just assume indices passed or handle error?
             # For simpler app, just assume preds is indices?
             # Let's keep consistent with model output.
             # If ndim is not 3, maybe we have (N, T) already?
             if logits.ndim == 2:
                 preds = logits.detach().cpu().numpy()
             else:
                 raise ValueError(f"Logits must be 2D (indices) or 3D (logits). Got {logits.ndim}")

        decoded_batch = []
        for sequence in preds:
            decoded_text = []
            prev_char = -1
            for char_idx in sequence:
                if char_idx != prev_char:
                    if char_idx != self.blank_idx:
                        decoded_text.append(self.int_to_char[char_idx])
                prev_char = char_idx
            decoded_batch.append("".join(decoded_text))
        return decoded_batch

def compute_cer(predicted: str, target: str) -> float:
    import editdistance
    return editdistance.eval(predicted, target) / max(len(target), 1)

def compute_wer(predicted: str, target: str) -> float:
    import editdistance
    pred_words = predicted.split()
    target_words = target.split()
    return editdistance.eval(pred_words, target_words) / max(len(target_words), 1)
