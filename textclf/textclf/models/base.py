import torch
import torch.nn as nn


class BaseDeepModel(nn.Module):
    def __init__(self):
        super(BaseDeepModel, self).__init__()

    def load_pretrained_embedding(self, pretrained_embedding_matrix):
        tmp = torch.FloatTensor(pretrained_embedding_matrix.float())
        if self.use_cuda:
            tmp = tmp.cuda()
        self.encoder_embedding.weight.data.copy_(tmp)
        self.decoder_embedding.weight.data.copy_(tmp)

    def flatten_parameters(self):
        pass

    def run_batch(self, batch):
        inps, inps_lens = batch.inps
        bert_inps, bert_inps_lens = batch.bert_inps
        labels = batch.labels
        logits = self.forward(
            inps=inps, inps_len=inps_lens,
            bert_inps=bert_inps, bert_inps_len=bert_inps_lens)

        loss = self.loss_fn(logits, labels)
        _, pred = logits.max(1)
        num_correct = pred.eq(labels).sum().item()
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words,
            "pred": pred.data.cpu().numpy().tolist()
        }
        return result_dict

    def predict_batch(self, batch):
        inps, inps_lens = batch.inps
        bert_inps, bert_inps_lens = batch.bert_inps
        logits = self.forward(
            inps=inps, inps_len=inps_lens,
            bert_inps=bert_inps, bert_inps_len=bert_inps_lens)
        _, pred = logits.max(1)
        return pred