"""
"""

import os
import copy
import argparse

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--yaml', '-y', type=str, default='cc-pg', help='configuration file path.')
_ARG_PARSER.add_argument('--cuda', '-c', type=str, default='0', help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--test', '-t', type=bool, default=False, help='只进行测试')
_ARG_PARSER.add_argument('--out', '-o', type=bool, default=False, help='预测结果输出')
_ARG_PARSER.add_argument('--name', '-n', type=str, default=None, help='save name.')
_ARG_PARSER.add_argument('--seed', '-s', type=int, default=123, help='random seed')
_ARG_PARSER.add_argument('--debug', '-d', default=False, action="store_true")
_ARG_PARSER.add_argument('--cache', type=bool, default=False, help='cache data')

_ARG_PARSER.add_argument('--adapter_size', type=int, default=None)
_ARG_PARSER.add_argument('--adapter_num', type=int, default=None)
_ARG_PARSER.add_argument('--lstm_size', type=int, default=None)
_ARG_PARSER.add_argument('--worker_dim', type=int, default=None)
_ARG_PARSER.add_argument('--pgn_layers', type=int, default=None)
_ARG_PARSER.add_argument('--share_param', type=bool, default=None)
_ARG_PARSER.add_argument('--extra_gold', type=float, default=None)
_ARG_PARSER.add_argument('--exclude_bad', type=bool, default=True)
_ARG_PARSER.add_argument('--start', type=int, default=None)
_ARG_PARSER.add_argument('--lr', type=float, default=None)

_ARGS = _ARG_PARSER.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda

if _ARGS:
    import random

    import numpy as np
    import torch

    from transformers import BertTokenizer, XLMTokenizer

    import basenlp
    from basenlp.common.config import load_yaml
    from basenlp.common.util import output, cache_path, load_cache, dump_cache
    from basenlp.common.writer import Writer
    from basenlp.core import Trainer, Vocabulary
    from basenlp.core.optim import build_optimizer

    from util import allowed_transition
    from models import build_model
    from datasets import CoNLL03Crowd, BAD_AID
else:
    raise Exception('Argument error.')

SEEDS = (123, 456, 789, 686, 666, 233, 1024, 2080, 3080, 3090)

basenlp.core.trainer.EARLY_STOP_THRESHOLD = 5


def set_seed(seed: int = 123):
    output(f"Process id: {os.getpid()}, cuda: {_ARGS.cuda}, set seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)  # CPU占用过高，但训练速度没快，还没找到问题所在


def run_once(cfg, dataset, vocab, device, writer=None, seed=123):
    model = build_model(vocab=vocab, **cfg.model)
    setattr(model, 'seed', seed)
    para_num = sum([p.numel() for p in model.parameters()])
    output(f'param num: {para_num}, {para_num / 1000000:4f}M')
    model.to(device=device)

    optimizer = build_optimizer(model, **cfg.optim)
    trainer = Trainer(vars(cfg), dataset, vocab, model, optimizer, None, None,
                      writer, device, **cfg.trainer)

    if not _ARGS.test:
        # 训练过程
        trainer.train()
        output(model.metric.data_info)

    trainer.load()
    if hasattr(dataset, 'kept'):
        dataset.kept.index_with(vocab)
        from util import scores_by_worker, save_csv
        scores = scores_by_worker()
        info_train, info_test = list(), list()
        for a in range(1, 48):
            r = [a, scores[a][2]]
            setattr(model, 'worker', a)
            train_a = copy.copy(dataset.train)
            train_a.data = [i for i in dataset.train.data if i['aid'] == a]
            output("ann: ", a, ", train num: ", len(train_a))
            m = trainer.test(train_a)
            r.append(len(train_a))
            r.append(m['main_F1'])

            test_a = copy.copy(dataset.kept)
            test_a.data = [i for i in test_a.data if i['aid'] == a]
            output("ann: ", a, ", kept num: ", len(test_a))
            m = trainer.test(test_a)
            r.append(len(test_a))
            r.append(m['main_F1'])

            output("ann: ", a, ", test ", len(dataset.test))
            m = trainer.test(dataset.test)
            r.append(m['main_F1'])

            r_train, r_test = copy.copy(r), copy.copy(r)
            for w in range(1, 48):
                train_i = copy.copy(dataset.train)
                train_i.data = [i for i in train_i.data if i['aid'] == w]
                m = trainer.test(train_i)
                r_train.append(m['main_F1'])

                test_i = copy.copy(dataset.kept)
                test_i.data = [i for i in test_i.data if i['aid'] == w]
                m = trainer.test(test_i)
                r_test.append(m['main_F1'])
            info_train.append(r_train)
            info_test.append(r_test)
            print('\n')
        setattr(model, 'worker', None)
        head = ['worker', 'f1-gold', 'train_num', 'train_f1', 'test_num', 'test_f1', 'conll_test_f1']
        head += [f"at_{wid}" for wid in range(1, 48)]
        save_csv([head] + info_test, 'dev/info_test.csv')
        save_csv([head] + info_train, 'dev/info_train.csv')

    test_metric = trainer.test(dataset.test)
    return model.metric.best, test_metric


def main(seed):
    cfg = argparse.Namespace(**load_yaml(f"./dev/config/{_ARGS.yaml}.yml"))

    device = torch.device("cuda:0")
    data_kwargs, vocab_kwargs = dict(cfg.data), dict(cfg.vocab)
    use_bert = 'bert' in cfg.model['word_embedding']['name_or_path']
    use_xlm = 'xlm' in cfg.model['word_embedding']['name_or_path']

    # 如果用了BERT，要加载tokenizer
    if use_bert:
        tokenizer = BertTokenizer.from_pretrained(
            cfg.model['word_embedding']['name_or_path'],
            do_lower_case=True)
        print("I'm batman!  ",
              tokenizer.tokenize("I'm batman!"))  # [CLS] [SEP]
        data_kwargs['tokenizer'] = tokenizer
        vocab_kwargs['oov_token'] = tokenizer.unk_token
        vocab_kwargs['padding_token'] = tokenizer.pad_token
    else:
        tokenizer = None

    cache_name = _ARGS.yaml
    prefix = _ARGS.name if _ARGS.name else _ARGS.yaml
    if _ARGS.extra_gold is not None:
        cache_name += f"-g{_ARGS.extra_gold}"
        prefix += f"-g{_ARGS.extra_gold}"
        cfg.data['extra_gold'] = _ARGS.extra_gold
    if _ARGS.exclude_bad:
        cache_name += "-exclude_bad"
        prefix += "-exclude_bad"
        cfg.data['exclude_bad'] = _ARGS.exclude_bad
        cfg.model['bad_ids'] = [0] + list(BAD_AID)

    if not os.path.exists(cache_path(cache_name)):
        dataset = argparse.Namespace(
            **CoNLL03Crowd.build(**cfg.data, tokenizer=tokenizer))
        vocab = Vocabulary.from_data(dataset, **vocab_kwargs)
        vocab.set_field(['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC'], 'tags')

        if use_bert:
            # 若用BERT，则把words词表替换为BERT的
            vocab.token_to_index['words'] = tokenizer.vocab
            vocab.index_to_token['words'] = tokenizer.ids_to_tokens
        if _ARGS.cache:
            dump_cache((dataset, vocab), cache_name)
    else:
        dataset, vocab = load_cache(cache_name)

    dataset.train.index_with(vocab)
    dataset.dev.index_with(vocab)
    dataset.test.index_with(vocab)

    if '-dis' in _ARGS.yaml:
        cfg.trainer['batch_size'] = 32
    cfg.model['allowed'] = allowed_transition(vocab)
    cfg.model['output_prediction'] = _ARGS.out

    if _ARGS.debug:
        log_dir = None
        cfg.trainer['save_strategy'] = 'no'
    else:
        # log_dir = f"./dev/tblog/{prefix}"
        # if not os.path.exists(log_dir):
        #     os.mkdir(log_dir)
        log_dir = None

    for k in ('lstm_size', 'adapter_size', 'adapter_num', 'pgn_layers', 'worker_dim'):
        p = getattr(_ARGS, k)
        if p is not None:
            cfg.model[k] = p
            prefix += f'-{k}{p}'

    if _ARGS.share_param:
        cfg.model['share_param'] = _ARGS.share_param
        prefix += '-share'
    if isinstance(_ARGS.start, int):
        cfg.model['path'] = f"./dev/model/cc-pg_{seed}_{_ARGS.start}.pth"
        prefix += f'-start{_ARGS.start}'
    if isinstance(_ARGS.lr, float):
        cfg.optim['lr'] = _ARGS.lr
        prefix += f'-lr{_ARGS.lr}'

    cfg.trainer['prefix'] = f"{prefix}_{seed}"
    if 'pre_train_path' not in cfg.trainer:
        cfg.trainer['pre_train_path'] = os.path.normpath(
            f"./dev/model/{cfg.trainer['prefix']}_best.pth")
    writer = Writer(log_dir, str(seed), 'tensorboard') if log_dir else None
    return run_once(cfg, dataset, vocab, device, writer, seed)


if __name__ == "__main__":
    set_seed(_ARGS.seed)
    main(_ARGS.seed)
