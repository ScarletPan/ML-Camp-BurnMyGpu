import torch
import torch.nn as nn
import torch.nn.functional as F
from textsum.models.base import BaseDeepModel
from textsum.toolbox.beam import BeamSeqs
from textsum.toolbox.layers import SortedLSTM, Attention
from textsum.toolbox.utils import batch_unpadding


class CopyEncoderDecoder(BaseDeepModel):
    def __init__(self, loss_fn, opt):
        super(CopyEncoderDecoder, self).__init__()
        rnn_hidden_size = opt.model.rnn_hidden_size

        self.encoder_embedding = nn.Embedding(opt.model.word_vocab_size, opt.model.word_embed_size)
        self.decoder_embedding = self.encoder_embedding
        self.encoder = SortedLSTM(input_size=opt.model.word_embed_size,
                                 hidden_size=opt.model.rnn_hidden_size // 2,
                                 num_layers=opt.model.n_layers,
                                 batch_first=True,
                                 bidirectional=True)
        self.decoder = SortedLSTM(input_size=opt.model.word_embed_size,
                                 hidden_size=opt.model.rnn_hidden_size,
                                 num_layers=opt.model.n_layers,
                                 batch_first=True,
                                 bidirectional=False)
        self.attn = Attention(input_size=opt.model.rnn_hidden_size, method=opt.model.attn_score_method)
        self.dropout = nn.Dropout(opt.model.dropout)
        self.concat = nn.Linear(rnn_hidden_size * 2, rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, opt.model.word_vocab_size)

        self.pgen = nn.Sequential(
            nn.Linear(rnn_hidden_size, 1),
            nn.Sigmoid())

        self.loss_fn = loss_fn
        self.rnn_hidden_size = opt.model.rnn_hidden_size
        self.n_layers = opt.model.n_layers
        self.use_cuda = opt.meta.use_cuda
        self.word_vocab_size = opt.model.word_vocab_size

    def encode(self, encoder_inputs, encoder_lens):
        encoder_embeds = self.encoder_embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(
            encoder_embeds, encoder_lens)
        return encoder_outputs, last_hidden

    def combine_probs(self, vocab_probs, attn_probs):
        """
        :param vocab_probs: bsize x vocab_size
        :param attn_probs: bsize x enc_seq_len
        :return:
        """

    def decode(self, encoder_outputs, encoder_lens, encoder_last_hidden,
               decoder_inputs, decoder_lens,
               ext_encoder_inputs, ext_vocab_size):
        bsize, enc_seq_len, _ = encoder_outputs.size()
        _, dec_seq_len = decoder_inputs.size()
        decoder_embeds = self.decoder_embedding(decoder_inputs)
        decoder_outputs, last_hidden = self.decoder(
            decoder_embeds, decoder_lens, encoder_last_hidden)

        contexts, attn_weights = self.attn(decoder_outputs, encoder_outputs,
                                           q_lens=decoder_lens, k_lens=encoder_lens)
        pgens = self.pgen(decoder_outputs)

        # Generate vocabulary dists
        outlayer_inputs = torch.cat([decoder_outputs, contexts], dim=2)
        outlayer_outputs = torch.tanh(self.concat(outlayer_inputs))
        vocab_dists = self.fc(outlayer_outputs)
        vocab_dists = F.softmax(vocab_dists, dim=2)
        vocab_dists = pgens.repeat(1, 1, vocab_dists.size(-1)) * vocab_dists

        if ext_vocab_size:
            new_vocab_size = bsize, dec_seq_len, ext_vocab_size
            ext_vocab_dist = decoder_inputs.data.new(*new_vocab_size).zero_().float() + 1.0 / self.word_vocab_size
            old_vocab_dists = torch.cat([vocab_dists, ext_vocab_dist], dim=2)
        else:
            old_vocab_dists = vocab_dists

        # # Attention dists
        attn_dists = (1 - pgens).repeat(1, 1, enc_seq_len) * attn_weights
        #
        new_vocab_dists = old_vocab_dists.scatter_add(
            2, ext_encoder_inputs.unsqueeze(1).repeat(1, dec_seq_len, 1), attn_dists)
        outputs = torch.log(new_vocab_dists + 1e-8)
        # outputs = new_vocab_dists
        return outputs, last_hidden, pgens, attn_dists

    def forward(self, encoder_inputs, encoder_lens,
                decoder_inputs, decoder_lens,
                ext_encoder_inputs, ext_vocab_size):
        encoder_outputs, encoder_last_hidden = self.encode(
            encoder_inputs, encoder_lens)
        encoder_last_hidden = (self._fix_hidden(encoder_last_hidden[0]),
                               self._fix_hidden(encoder_last_hidden[1]))
        outputs, _, pgens, attn_dists = self.decode(
            encoder_outputs, encoder_lens, encoder_last_hidden,
            decoder_inputs, decoder_lens,
            ext_encoder_inputs, ext_vocab_size)
        return outputs, pgens, attn_dists

    def generate(self, encoder_inputs, encoder_lens,
                 decoder_start_input, max_len,
                 ext_encoder_inputs, ext_vocab_size, beam_size=1, eos_val=None):
        encoder_outputs, encoder_last_hidden = self.encode(encoder_inputs, encoder_lens)
        encoder_last_hidden = (self._fix_hidden(encoder_last_hidden[0]),
                               self._fix_hidden(encoder_last_hidden[1]))
        beamseqs = BeamSeqs(beam_size=beam_size)
        beamseqs.init_seqs(seqs=decoder_start_input[0], init_state=encoder_last_hidden)
        done = False

        for i in range(max_len):
            for j, (seqs, _, last_token, last_hidden, *_) in enumerate(beamseqs.current_seqs):
                if beamseqs.check_and_add_to_terminal_seqs(j, eos_val):
                    if len(beamseqs.terminal_seqs) >= beam_size:
                        done = True
                        break
                    continue
                if last_token.item() >= self.word_vocab_size:
                    last_token = last_token * 0 + 1
                out, last_hidden, pgens, attn_dists = self.decode(
                    encoder_outputs, encoder_lens, last_hidden,
                    last_token.unsqueeze(0), None,
                    ext_encoder_inputs, ext_vocab_size)
                _output = out.squeeze(0).squeeze(0)
                scores, tokens = _output.topk(beam_size * 2)
                for k in range(beam_size * 2):
                    score, token = scores.data[k], tokens[k]
                    token = token.unsqueeze(0)
                    beamseqs.add_token_to_seq(j, token, score, last_hidden, attn_dists, pgens)
            if done:
                break
            beamseqs.update_current_seqs()
        final_seqs = beamseqs.return_final_seqs()
        attns = torch.cat([t for t in final_seqs[-2]], dim=1).data.cpu().numpy().tolist()
        pgns = torch.cat([t.squeeze(2) for t in final_seqs[-1]], dim=1).data.cpu().numpy().tolist()
        return final_seqs[0].unsqueeze(0).squeeze(2), attns, pgns

    @staticmethod
    def _fix_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                            hidden[1:hidden.size(0):2]], 2)
        return hidden

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def run_batch(self, batch):
        enc_inps, enc_lens = batch.enc_inps
        dec_inps, dec_lens = batch.dec_inps
        dec_tgts, _ = batch.dec_tgts
        dec_probs, pgens, attn_dists = self.forward(
            encoder_inputs=enc_inps, encoder_lens=enc_lens,
            decoder_inputs=dec_inps, decoder_lens=dec_lens,
            ext_encoder_inputs=batch.ext_enc_inps[0],
            ext_vocab_size=batch.max_ext_vocab_size)

        # ext_enc_inps = batch.ext_enc_inps[0]
        # ext_vocab = batch.oov_vocabs
        # for i in range(dec_inps.size(0)):
        #     dec_idxs = dec_tgts[i].data.cpu().numpy()
        #     for j, oov_idx in enumerate(dec_idxs):
        #         if ext_vocab[i].size > 0 and oov_idx > 50000:
        #             enc_idxs = ext_enc_inps[i].data.cpu().numpy()
        #             for k, enc_idx in enumerate(enc_idxs):
        #                 if enc_idx == oov_idx:
        #                     print("Encoder oov:")
        #                     print(ext_enc_inps[i, k])
        #                     print(attn_dists[i, j, k])
        #             print("Decoder oov:")
        #             print(j, oov_idx, dec_tgts[i, j])
        #             print(ext_vocab[i].stoi)
        #             print(pgens[i, j])
        #             print(dec_probs[i, j, oov_idx])
        #             print("="*50)

        decoder_probs_pack = dec_probs.view(-1, dec_probs.size(2))
        decoder_targets_pack = dec_tgts.view(-1)
        loss = self.loss_fn(decoder_probs_pack, decoder_targets_pack)
        decoder_probs_pack = batch_unpadding(dec_probs, dec_lens)
        decoder_targets_pack = batch_unpadding(dec_tgts, dec_lens)
        _, pred = decoder_probs_pack.max(1)
        num_correct = pred.eq(decoder_targets_pack).sum().item()
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words,
        }
        return result_dict

    def predict_batch(self, batch, max_len=20, beam_size=4, eos_val=0):
        enc_inps, enc_lens = batch.enc_inps
        dec_start_inps = batch.dec_start_inps
        preds, attns, pgns = self.generate(enc_inps, enc_lens,
                              dec_start_inps, max_len,
                              batch.ext_enc_inps[0],
                              batch.max_ext_vocab_size,
                              beam_size, eos_val)
        preds = preds.data.cpu().numpy()
        return preds, attns, pgns