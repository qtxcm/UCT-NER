import os
import shutil
from typing import Dict, Tuple, List, Any, cast, Union

import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterList, init

from basenlp.core import Vocabulary
from basenlp.core.trainer import format_metric, output
from basenlp.embedding import build_word_embedding
from basenlp.modules.adapter import AdapterBertModel
from basenlp.modules.dropout import WordDropout
from basenlp.modules.encoder import LstmEncoder

from metric import ExactMatch
from conditional_random_field import ConditionalRandomField


def build_model(name, **kwargs):
    m = {
        'tag': Tagger,
        'ad': AdapterModel,
        'pg': PGNModel,
        'lstm': LSTMCrowd,
        'ft': Finetune,
        'gold': GoldFineTune,
        'adft': AdaFineTune,
        'lcft': LCFineTune
    }
    return m[name](**kwargs)


def tensor_like(data, t: torch.Tensor) -> torch.Tensor:
    tensor = torch.zeros_like(t)
    for i, l in enumerate(data):
        tensor[i, :len(l)] = torch.tensor(l, dtype=t.dtype, device=t.device)
    return tensor


class CRF(nn.Module):
    """ CRF classifier."""
    def __init__(
        self,
        num_tags: int,
        input_dim: int = 0,
        top_k: int = 1,
        reduction='mean',
        constraints: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True
    ) -> None:
        super().__init__()
        if input_dim > 0:
            self.tag_projection = nn.Linear(input_dim, num_tags)
        else:
            self.tag_projection = None

        self.base = ConditionalRandomField(
            num_tags, reduction, constraints, include_start_end_transitions) #CRF
        self.top_k = top_k

    def forward(
        self, inputs: torch.FloatTensor, mask: torch.LongTensor,
        labels: torch.LongTensor = None, reduction: str = None,
    ) -> Dict[str, Any]:
        bool_mask = mask.bool()
        if self.tag_projection:
            inputs = self.tag_projection(inputs)
        scores = inputs * mask.unsqueeze(-1)

        best_paths = self.base.viterbi_tags(scores, bool_mask, top_k=self.top_k)
        # Just get the top tags and ignore the scores.
        tags = cast(List[List[int]], [x[0][0] for x in best_paths])

        if labels is None:
            loss = torch.tensor(0, dtype=torch.float, device=inputs.device)
        else:
            # Add negative log-likelihood as loss
            loss = self.base(scores, labels, bool_mask, reduction)

        return dict(scores=scores, predicted_tags=tags, loss=loss)


class Tagger(nn.Module):
    """ a """
    def __init__(
        self,
        vocab: Vocabulary,
        word_embedding: Dict[str, Any],
        lstm_size: int = 400,
        input_namespace: str = 'words',
        label_namespace: str = 'tags',
        save_embedding: bool = False,
        allowed: List[Tuple[int, int]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.word_embedding = build_word_embedding(
            num_embeddings=vocab.size_of(input_namespace), vocab=vocab, **word_embedding)

        if lstm_size > 0:
            self.lstm = LstmEncoder(
                self.word_embedding.output_dim, lstm_size,
                num_layers=1, bidirectional=True)
            feat_dim = self.lstm.output_dim
        else:
            self.lstm = lambda *args: args[0]
            feat_dim = self.word_embedding.output_dim

        num_tags = vocab.size_of(label_namespace)
        self.tag_projection = nn.Linear(feat_dim, num_tags)

        self.word_dropout = WordDropout(0.20)
        self.crf = CRF(num_tags, constraints=allowed)
        self.metric = ExactMatch(
            vocab.index_of('O', label_namespace),
            vocab.token_to_index[label_namespace],
            True, True)
        self.save_embedding = save_embedding
        self.id_to_label = vocab.index_to_token[label_namespace]
        self.epoch = 0

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor,
        mask: torch.Tensor = None, tags: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        embedding = self.word_embedding(words, mask=mask,**kwargs)
        embedding = self.word_dropout(embedding)
        feature = self.lstm(embedding, lengths)
        scores = self.tag_projection(feature)
        output_dict = self.decode(scores, mask, tags, lengths)
        return output_dict

    def decode(self, scores, mask, labels, lengths):
        output_dict = self.crf(scores, mask, labels)
        if labels is not None:
            output_dict = self.add_metric(output_dict, labels, lengths)
        return output_dict

    def add_metric(self, output_dict, tags, lengths, prefix=''):
        prediction = tensor_like(output_dict['predicted_tags'], tags)
        output_dict['metric'] = getattr(self, prefix + "metric")(prediction, tags, lengths)
        return output_dict

    def before_train_once(self, kwargs):
        self.epoch = kwargs['epoch']

    def after_epoch_end(self, kwargs):
        output(f"Train {format_metric(kwargs['metric'])}")

    def save(self, path):
        state_dict = self.state_dict()
        if not self.save_embedding:
            state_dict = {k: v for k, v in state_dict.items() if not self.drop_param(k)}
        torch.save(state_dict, path)

    def load(self, path_or_state, device):
        if isinstance(path_or_state, str):
            path_or_state = torch.load(path_or_state, map_location=device)
        # if 'woker_matrix.weight' in path_or_state:
        #     path_or_state.pop('woker_matrix.weight')
        info = self.load_state_dict(path_or_state, strict=False)
        missd = [i for i in info[0] if not self.drop_param(i)]
        if missd:
            print(missd)
        # print("model loaded.")

    def drop_param(_, name: str):
        return name.startswith('word_embedding.bert')


class AdapterModel(Tagger):
    def __init__(self,
                 vocab: Vocabulary,
                 adapter_size: int = 128,
                 external_param: Union[bool, List[bool]] = False,
                 adapter_num: int = 12,
                 output_prediction: bool = False,
                 **kwargs):
        super().__init__(vocab, **kwargs)
        self.word_embedding = AdapterBertModel(
            self.word_embedding.bert, adapter_size, adapter_num, external_param)
        self.output_prediction = output_prediction
        # self.out_dir = 'dev/out/cc-mv-semi-rep_123/'

    def drop_param(_, name: str):
        return super().drop_param(name) and 'LayerNorm' not in name

    def before_time_start(self, _, trainer, kwargs):
        if self.output_prediction:
            out_dir = f"dev/out/{trainer.prefix}/"
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.mkdir(out_dir)
            setattr(self, 'out_dir', out_dir)

    def before_next_batch(self, kwargs):
        if self.training or not self.output_prediction:
            return
        epoch, batch, out = kwargs['epoch'], kwargs['batch'], kwargs['output_dict']
        name = f"{'test' if epoch is None else 'dev'}-{self.epoch}"
        if hasattr(self, 'worker'):
            name += f"-w{self.worker}"
        with open(f"{self.out_dir}/{name}.txt", mode='a') as file:
            for ins, pred in zip(batch, out['predicted_tags']):
                ziped = list(zip(ins['text'], ins['tags'], pred))[1:-1]
                for w, li, pi, in ziped:
                    file.write(f"{w}\t{self.id_to_label[li]}\t{self.id_to_label[pi]}\n")
                file.write('\n')
        return


class PGNModel(AdapterModel):
    def __init__(self,
                 vocab: Vocabulary,
                 worker_dim: int = 8,
                 worker_num: int = 48,
                 adapter_size: int = 128,
                 pgn_layers: int = 12,
                 share_param: bool = False,
                 worker: int = None,
                 crowd_test: bool = False,
                 bad_ids=(0,),
                 **kwargs):
        super().__init__(vocab, adapter_size, [True] * pgn_layers, **kwargs)
        self.worker_embedding = nn.Embedding(worker_num, worker_dim)  # max_norm=1.0
        dim = self.word_embedding.output_dim
        size = [2] if share_param else [pgn_layers, 2]
        self.weight = ParameterList([
            Parameter(torch.Tensor(*size, adapter_size, dim, worker_dim)),
            Parameter(torch.zeros(*size, adapter_size, worker_dim)),
            Parameter(torch.Tensor(*size, dim, adapter_size, worker_dim)),
            Parameter(torch.zeros(*size, dim, worker_dim)),
        ])
        self.reset_parameters()
        self.adapter_size = adapter_size
        self.pgn_layers = pgn_layers
        self.share_param = share_param
        self.worker = worker
        self.crowd_test = crowd_test
        self.mean_ids = [i for i in range(48) if i not in bad_ids]

    def reset_parameters(self):
        # bound = 1e-2
        # init.uniform_(self.weight[0], -bound, bound)
        # init.uniform_(self.weight[2], -bound, bound)
        init.normal_(self.weight[0], std=1e-3)
        init.normal_(self.weight[2], std=1e-3)

    def set_worker(self, aid: torch.LongTensor):
        if self.training or self.crowd_test:
            embedding = self.worker_embedding(aid)
        elif isinstance(self.worker, int):
            embedding = self.worker_embedding.weight[self.worker]
        else:
            ids = torch.tensor(self.mean_ids, device=self.worker_embedding.weight.device)
            embedding = self.worker_embedding(ids).mean(0)
        self.set_adapter_parameter(embedding)

    def set_adapter_parameter(self, embedding: torch.Tensor):
        def batch_matmul(w: torch.Tensor, e):
            ALPHA = "ijklmnopqrstuvwxyz"
            dims = ALPHA[:w.dim() - 1]
            i = 1 if self.share_param else 2
            return torch.einsum(f"{dims}a,ba->{dims[:i] + 'b' + dims[i:]}", w, e)

        matmul = batch_matmul if embedding.dim() == 2 else torch.matmul
        embedding = embedding.softmax(-1)
        param_list = [matmul(w, embedding) for w in self.weight]

        for i, adapters in enumerate(self.word_embedding.adapters[-self.pgn_layers:]):
            for j, adapter in enumerate(adapters):
                params: List[torch.Tensor] = [p[j] if self.share_param else p[i, j] for p in param_list]
                setattr(adapter, 'params', params)

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor,
        aid: torch.LongTensor = None, tags: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        self.set_worker(aid)
        return super().forward(words, lengths, mask, tags, **kwargs)


class GoldFineTune(PGNModel):
    def __init__(
        self, vocab: Vocabulary, path: str, **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.path = path

    def before_time_start(self, dataset, trainer, kwargs):
        super().before_time_start(dataset, trainer, kwargs)
        print('load ', self.path)
        self.load(self.path, trainer.device)
        print('avg')
        trainer.test(dataset.test)
        print('expert')
        self.worker = 0
        trainer.test(dataset.test)


class AdaFineTune(AdapterModel):
    def __init__(
        self, vocab: Vocabulary, path: str, **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.path = path

    def before_time_start(self, dataset, trainer, kwargs):
        super().before_time_start(dataset, trainer, kwargs)
        print('load ', self.path)
        self.load(self.path, trainer.device)
        print('test')
        trainer.test(dataset.test)


class LSTMCrowd(AdapterModel):
    def __init__(self,
                 vocab: Vocabulary,
                 worker_num: int = 48,
                 label_namespace: str = 'tags',
                 cat=False,
                 crowd_test: bool = False,
                 **kwargs):
        super().__init__(vocab, label_namespace=label_namespace, **kwargs)
        self.cat = cat
        label_num = vocab.size_of(label_namespace)
        if cat:
            self.tag_projection = nn.Linear(self.lstm.output_dim * 2, label_num)
            self.woker_matrix = nn.Embedding(
                worker_num, self.lstm.output_dim, padding_idx=0,
                _weight=torch.zeros(worker_num, self.lstm.output_dim))
        else:
            self.woker_matrix = nn.Embedding(
                worker_num, label_num, _weight=torch.zeros(worker_num, label_num))
        self.crowd_test = crowd_test

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor,
        aid: torch.Tensor = None, tags: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        with torch.no_grad():  # set expert to 0
            self.woker_matrix.weight[0].zero_()

        embedding = self.word_embedding(words, mask=mask, **kwargs)
        embedding = self.word_dropout(embedding)
        feature = self.lstm(embedding, lengths)

        if self.training or (self.crowd_test and aid[0].item() != 0):
            vector = self.woker_matrix(aid).unsqueeze(1).expand(-1, words.size(1), -1)
        elif hasattr(self, 'worker'):
            vector = self.woker_matrix.weight[self.worker]
            vector = vector.unsqueeze(0).unsqueeze(0).expand(words.size(0), words.size(1), -1)
        else:
            vector = torch.zeros_like(feature)

        if self.cat:
            feature = torch.cat([feature, vector], dim=-1)

        scores = self.tag_projection(feature)
        if not self.cat and (self.training or (self.crowd_test and aid[0].item() != 0) or hasattr(self, 'worker')):
            scores += vector

        output_dict = self.decode(scores, mask, tags, lengths)
        return output_dict


class LCFineTune(LSTMCrowd):
    def __init__(
        self, vocab: Vocabulary, path: str, **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.path = path

    def before_time_start(self, dataset, trainer, kwargs):
        super().before_time_start(dataset, trainer, kwargs)
        print('load ', self.path)
        state = torch.load(self.path, map_location=trainer.device)
        # _ = state.pop('woker_matrix.weight')
        info = self.load_state_dict(state, strict=False)
        missd = [i for i in info[0] if not self.drop_param(i)]
        if missd:
            print(missd)
        # with torch.no_grad():
        #     for i in range(vec.size(0)):
        #         self.woker_matrix[i + 1] = vec[i]
        print('test')
        trainer.test(dataset.test)


class Finetune(Tagger):
    def __init__(self,
                 vocab: Vocabulary,
                 **kwargs):
        super().__init__(vocab, lstm_size=0, **kwargs)
        for param in self.word_embedding.parameters():
            param.requires_grad = True

    def drop_param(_, name: str):
        return False
