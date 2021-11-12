import Levenshtein as Lev
import torch


class GreedyDecoder:
    def __init__(self, tokenizer, blank_id, space_symbol='_'):
        self.tokenizer = tokenizer
        self.lables = tokenizer.vocab()
        self.blank_id = blank_id
        self.space_symbol = space_symbol
        if not self.space_simbol in labels:
            raise ValueError(f"there is not space symbol: {self.space_symbol}")

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append(string)  # We only return one path
            if return_offsets:
                offsets.append(string_offsets)
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char(sequence[i].item())
            if char != self.int_to_char(self.blank_index):
                if remove_repetitions and i != 0 and char == self.int_to_char(sequence[i - 1].item()):
                    pass
                elif char == self.labels[self.space_index]:
                    string += self.space_simbol
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)
