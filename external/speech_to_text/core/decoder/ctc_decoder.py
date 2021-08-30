from .base_decoder import BaseDecoder
from ctcdecode import CTCBeamDecoder


class CTCDecoder(BaseDecoder):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.decoder = CTCBeamDecoder(
            list(self.tokenizer.vocab),
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=100,
            num_processes=4,
            blank_id=self.tokenizer.get_blank_symbol()[1],
            log_probs_input=False
        )

    def __call__(self, probs, sizes=None):
        return self.decode(probs, sizes)

    def decode(self, probs, sizes=None):
        out, scores, offsets, seq_lens = self.decoder.decode(probs, sizes)
        strings = self._convert_to_strings(out, seq_lens)
        strings = [item[0] for item in strings]
        return strings

    def _convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.tokenizer.id_to_subword(x.item()), utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results
