import torch


class BeamSeqs(object):
    def __init__(self, beam_size):
        self.current_seqs = []
        self.new_seqs = []
        self.terminal_seqs = []
        self.beam_size = beam_size

    def init_seqs(self, seqs, init_state):
        latest_token = seqs[-1]
        init_score = 0
        self.current_seqs.append((seqs, init_score, latest_token, init_state, [], []))

    def add_token_to_seq(self, i, token, new_score, last_hidden, attn=None, pgn=None):
        seq, score, _, _, attns_list, pgns_list = self.current_seqs[i]
        new_attns_list = attns_list[:]
        new_pgns_list = pgns_list[:]
        seq = torch.cat([seq, token.unsqueeze(0)])
        if attn is not None:
            new_attns_list.append(attn.data.cpu().numpy())
        if pgn is not None:
            new_pgns_list.append(pgn.data.cpu().numpy())
        self.new_seqs.append((seq, score + new_score, token, last_hidden,
                              new_attns_list, new_pgns_list))

    def update_current_seqs(self):
        self.current_seqs = self.new_seqs
        self.current_seqs = [item for item in self.current_seqs if item is not None]
        if len(self.current_seqs) > self.beam_size:
            self.current_seqs = sorted(
                self.current_seqs,
                key=lambda x: x[1], # / (((5 + x[0].size(0)) ** 0.6) / 6 ** 0.6),
                reverse=True)[:self.beam_size]
        self.new_seqs = []

    def check_and_add_to_terminal_seqs(self, j, eos_val):
        tmp = self.current_seqs[j]
        seqs = tmp[0]
        if seqs[-1].data[0] == eos_val:
            if seqs.size(0) >= 5:
                self.terminal_seqs.append(self.current_seqs[j])
            self.current_seqs[j] = None
            return True
        else:
            return False

    def return_final_seqs(self, K=1):
        candidate_seqs = []
        if len(self.terminal_seqs) > 0:
            candidate_seqs = self.terminal_seqs
        sorted_seqs = sorted(self.terminal_seqs, key=lambda x: x[1], reverse=True) # / (((5 + x[0].size(0)) ** 0.6) / 6 ** 0.6))
        if len(sorted_seqs) < K:
            tmp_sorted_seqs = sorted(self.current_seqs, key=lambda x: x[1],
                                 reverse=True)  # / (((5 + x[0].size(0)) ** 0.6) / 6 ** 0.6))
            sorted_seqs.extend(tmp_sorted_seqs[:K - len(candidate_seqs)])

        return sorted_seqs[:K]
