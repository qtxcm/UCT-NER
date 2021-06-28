"""
"""

from typing import Set, Tuple
from itertools import chain
from collections import OrderedDict

from torch import LongTensor

from basenlp.core.metrics import TaggingMetric, namespace_add


class ExactMatch(TaggingMetric):
    def __init__(self, o_id, token_to_id, ouput_class=False, output_detail=False):
        super().__init__(o_id)
        self.o_id = o_id
        self.id_to_label = dict()  # map[i_x] = x
        self.bi_map = dict()  # map[b_x] = i_x
        for label, index in token_to_id.items():
            if label.startswith('B-'):
                self.bi_map[label[2:]] = index
        for label, index in token_to_id.items():
            if label.startswith('I-'):
                b_id = self.bi_map.pop(label[2:])
                self.bi_map[b_id] = index
                self.id_to_label[index] = label[2:]

        self.label_counter = {k: self.counter_factory() for k in self.id_to_label}
        self.ouput_class = ouput_class
        self.output_detail = output_detail
        self.data_info = dict()

    def __call__(self,
                 predictions: LongTensor,
                 gold_labels: LongTensor,
                 lengths: LongTensor) -> OrderedDict:
        batch = self.counter_factory()

        for prediction, gold, length in zip(predictions, gold_labels, lengths):
            predict_entities = self.get_entities(prediction.tolist()[:length])
            gold_entities = self.get_entities(gold.tolist()[:length])
            correct_entities = self.get_correct(predict_entities, gold_entities)

            for e in gold_entities:
                self.label_counter[e[2]].total += 1
                batch.total += 1
            for e in predict_entities:
                self.label_counter[e[2]].positive += 1
                batch.positive += 1
            for e in correct_entities:
                self.label_counter[e[2]].correct += e[3]
                batch.correct += e[3]

        self.counter = namespace_add(self.counter, batch)

        return self.get_metric(batch)

    def get_entities(self, labels) -> Set[Tuple[int]]:
        entities, one = set(), None
        for i, label in enumerate(chain(labels, [self.o_id])):
            if one:
                if label == one[2]:  # I-x
                    one[1] = i
                    continue
                else:
                    entities.add(tuple(one))
                    one = None
            if label in self.bi_map:  # B-x
                one = [i, i, self.bi_map[label]]  # start, end, I-x
        return entities

    @staticmethod
    def get_correct(predict_entities, gold_entities):
        correct_entities = predict_entities & gold_entities
        correct_entities = {tuple(chain(e, [1])) for e in correct_entities}
        return correct_entities

    def get_metric(self, counter=None, reset=False) -> OrderedDict:
        if not reset:
            return super().get_metric(counter)
        if not self.ouput_class:
            return super().get_metric(reset=True)

        key_list = ['F1', 'precision', 'recall'] if self.output_detail else ['F1']

        metrics = dict(main=super().get_metric(reset=True))
        for k, counter in self.label_counter.items():
            self.data_info[self.id_to_label[k]] = counter.total
            metrics[k] = super().get_metric(counter)
            self.label_counter[k] = self.counter_factory()

        metric_with_prefix = OrderedDict()
        for prefix in chain(['main'], self.label_counter.keys()):
            for k in key_list:
                prefix_str = self.id_to_label[prefix] if isinstance(prefix, int) else prefix
                metric_with_prefix[f"{prefix_str}_{k}"] = metrics[prefix][k]

        return metric_with_prefix
