import csv
import random
import pickle
from typing import List, Tuple
# from difflib import SequenceMatcher
from itertools import chain
from collections import defaultdict

from basenlp.core import Vocabulary


def allowed_transition(vocab: Vocabulary, namespace='tags') -> List[Tuple]:
    def idx(token: str) -> int:
        return vocab.index_of(token, namespace)

    allowed, keys = list(), vocab.token_to_index[namespace].keys()
    for i in keys:
        for j in keys:
            if i == "O" and j.startswith("I-"):
                continue
            if i.startswith("B-") and j.startswith("I-") and i.split('-')[1] != j.split('-')[1]:
                continue
            if i.startswith("I-") and j.startswith("I-") and i.split('-')[1] != j.split('-')[1]:
                continue
            allowed.append((idx(i), idx(j)))

    return allowed


def read_lines(path, sep=" "):
    with open(path, mode='r', encoding='UTF-8') as conllu_file:
        sentence = list()
        for line in chain(conllu_file, [""]):
            line = line.strip()
            if not line and sentence:
                yield sentence
                sentence = list()
            elif line.startswith("-DOCSTART-"):
                continue
            else:
                cols = line.split(sep)
                if len(cols) > 1:
                    sentence.append(cols)
                else:
                    print(cols)


def write_lines(data, path, sep=' '):
    with open(path, mode='w', encoding='UTF-8') as file:
        for ins in data:
            for line in ins:
                file.write(sep.join(line) + '\n')
            file.write('\n')
    print('write to ', path)


def save_csv(rows, path='dev/metrics.csv'):
    with open(path, mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    print(f"saved at <{path}>")


def bio_to_span_t(tokens: List[str], seq: List[str]):
    spans = list()
    one = None  # [i, j, label, length]  如果切片要+1
    for i, (t, w) in enumerate(zip(chain(seq, ['O']), chain(tokens, ['_']))):
        if t.startswith("B-"):
            if one:
                spans.append(tuple(one))
            one = [i, i, t[2:], 1]
        elif t.startswith("I-"):
            if one is None:
                print("有I前面无B")
                # raise Exception("有I前面无B")
            elif one[2] != t[2:]:
                raise Exception("BI不一致")
            else:
                one[1] = i
                if not w.startswith('##'):
                    one[3] += 1
        elif one:
            spans.append(tuple(one))
            one = None

    return spans


def bio_to_span(seq: List[str]):
    spans = list()
    one = None  # [i, j, label]  如果切片要+1
    for i, t in enumerate(chain(seq, ['O'])):
        if t.startswith("B-"):
            if one:
                spans.append(tuple(one))
            one = [i, i, t[2:]]
        elif t.startswith("I-"):
            if one is None:
                print('有I前面无B')
                # raise Exception("有I前面无B")
            elif one[2] != t[2:]:
                print('BI不一致')
                # raise Exception("BI不一致")
            else:
                one[1] = i
        elif one:
            spans.append(tuple(one))
            one = None

    return spans


def sample_data():
    DIR = 'dev/data/conll03-crowd/'
    # crowd = list(read_lines(DIR + 'answers.txt'))
    # train, test = list(), list()
    # for line in crowd:
    #     if random.uniform(0, 1) >= 0.85:
    #         test.append(line)
    #     else:
    #         train.append(line)
    # write_lines(train, DIR + "answers-85.txt")
    # write_lines(test, DIR + "answers-15.txt")

    gold = list(read_lines(DIR + 'ground_truth.txt'))
    # conll = list(read_lines(DIR + 'train.bio'))
    # distant, matched = set(), set()
    # for i, ins in enumerate(conll):
    #     text = ''.join(c[0] for c in ins)
    #     for j, ing in enumerate(gold):
    #         if j in matched:
    #             continue
    #         sim = SequenceMatcher(None, text, ''.join(c[0] for c in ing)).quick_ratio()
    #         if sim > 0.98:
    #             matched.add(j)
    #             break
    #     else:
    #         distant.add(i)
    # conll_rest = [c for i, c in enumerate(conll) if i in distant]

    def filter(lines, tags=2, length=8) -> bool:
        if len(lines) < length:
            return False
        if sum(1 if 'B-' in i[1] else 0 for i in lines) < tags:
            return False
        return True

    with open('dev/data/rest.pkl', mode='rb') as f:
        conll_rest = pickle.load(f)

    def multi(data, name):
        good = set(i for i, lines in enumerate(data) if filter(lines, 3))
        print('good: ', len(good))
        p1 = set(random.sample(good, 60))
        p1d = [d for i, d in enumerate(data) if i in p1]
        print('prop: 0.01, num: ', len(p1d))
        write_lines(p1d, f"{DIR}{name}-0.01.txt")
        left = good.difference(p1)
        print('1 left: ', len(left))
        p5 = set(random.sample(left, 240))
        p5d = p1d + [d for i, d in enumerate(data) if i in p5]
        print('prop: 0.05, num: ', len(p5d))
        write_lines(p5d, f"{DIR}{name}-0.05.txt")
        left = left.difference(p5)
        print('5 left: ', len(left))

        two = set(i for i, lines in enumerate(data) if filter(lines, 2))
        two = two.difference(good)
        need = 1197 - len(left)
        if need > 0:
            print('need: ', need)
            p25 = left.union(set(random.sample(two, need)))
        else:
            p25 = set(random.sample(left, 1197))
            left = left.difference(p25)
            print('25 left: ', len(left))
        p25d = p5d + [d for i, d in enumerate(data) if i in p25]
        print('prop: 0.25, num: ', len(p25d))
        write_lines(p25d, f"{DIR}{name}-0.25.txt")
        if name == 'gold':
            print('prop: 1.0, num: ', len(data))
            p100d = data
        else:
            one = set(i for i, lines in enumerate(data) if filter(lines, 1, 2))
            one = one.difference(two).difference(good)
            print('one: ', len(one))
            ids = set(i for i, lines in enumerate(data) if filter(lines, 0, 4))
            ids = ids.difference(one).difference(two).difference(good)
            need = 5985 - len(p25d) - len(left) - len(one)
            print('need: ', need, ', ids: ', len(ids))
            p100 = left.union(set(random.sample(ids, need))).union(one)
            p100d = p25d + [d for i, d in enumerate(data) if i in p100]
            print('prop: 1.0, num: ', len(p100d))
        write_lines(p100d, f"{DIR}{name}-1.0.txt")
        return

    multi(gold, 'gold')
    print('\n')
    multi(conll_rest, 'rest')

    # for p in (0.01, 0.05, 0.25, 1.0):  60, 300, 1497, 5985
    #     num = int(ALL * p + 1)
    #     print('prop: ', p, ', num: ', num)
    #     sampled = random.sample(conll_rest, num)
    #     write_lines(sampled, f"{DIR}rest-{p}.txt")
    #     sampled = random.sample(gold, num)
    #     write_lines(sampled, f"{DIR}gold-{p}.txt")
    return


def scores_by_entity(path):
    data = read_lines(path, sep='\t')
    # total_gold, correct_gold, total_pred, correct_pred = 0, 0, 0, 0
    info = defaultdict(lambda: [0, 0, 0, 0])

    def add_info(length, i):
        if length > 3:
            info[4][i] += 1
        else:
            info[length][i] += 1

    for ins in data:
        tokens = [i[0] for i in ins]
        tags = [i[1] for i in ins]
        pres = [i[2] for i in ins]
        gold = bio_to_span_t(tokens, tags)
        for e in gold:
            add_info(e[3], 0)
        pred = bio_to_span_t(tokens, pres)
        for e in pred:
            add_info(e[3], 2)
        gc = [g for g in gold if g in pred]
        for e in gc:
            add_info(e[3], 1)
        pc = [p for p in pred if p in gold]
        for e in pc:
            add_info(e[3], 3)

    for k, v in info.items():
        p = v[3] / v[2]
        r = v[1] / v[0]
        f = 2 * p * r / (p + r)
        print(f"len {k}, ng: {v[0]}, np: {v[2]},  p: {p:.4f}, r: {r:.4f}, f: {f:.4f}")

    return ''


def scores_by_worker():
    # crowd = list(read_lines('data/answers.txt'))
    # gold = list(read_lines('data/ground_truth.txt'))
    # matched, mg = list(), set()
    # for c in crowd:
    #     ct = [i[0] for i in c]
    #     for gi, g in enumerate(gold):
    #         if gi in mg:
    #             continue
    #         cg = [i[0] for i in g]
    #         if ct == cg:
    #             mg.add(gi)
    #             break
    #     else:
    #         print(ct)
    #         continue
    #     for i, j in zip(c, g):
    #         i[0] = j[1]
    #     matched.append(c)

    with open('dev/data/match.pkl', mode='rb') as f:
        matched = pickle.load(f)

    info = defaultdict(lambda: [0, 0, 0])
    for ins in matched:
        gt = [i[0] for i in ins]
        egt = set(bio_to_span(gt))
        for w in range(1, 48):
            if ins[0][w] == '?':
                continue
            ti = [i[w] for i in ins]
            eti = set(bio_to_span(ti))
            cti = egt & eti
            info[w][0] += len(egt)
            info[w][1] += len(eti)
            info[w][2] += len(cti)

    scores = dict()
    for k, v in info.items():
        p = v[2] / v[1]
        r = v[2] / v[0]
        f = 2 * p * r / (p + r)
        scores[k] = [p, r, f]

    for k in sorted(scores, key=lambda x: scores[x][2]):
        p, r, f = scores[k]
        print(f"worker {k}, ng: {info[k][0]}, np: {info[k][1]},  p: {p:.4f}, r: {r:.4f}, f: {f:.4f}")

    return scores


def main():
    # sample_data()
    data = list()
    for p in (
        'dev/out/cc-gt/test.txt',
        'dev/out/cc-mv/test.txt',
        'dev/out/lc-cat_123/test-3.txt',
        'dev/out/cc-pg_123/test-4-wNone.txt',
        'dev/out/cc-mv-semi-g0.05_123/test-6.txt',
        'dev/out/lc-cat-semi-g0.05_123/test-3.txt',
        'dev/out/cc-pg-semi-g0.05_123/test-13-w0.txt',
    ):
        data.append(list(read_lines(p, sep='\t')))
        # print('\n', p)
        # print(scores_by_entity(p))
    predictions = list()
    for i in range(len(data[0])):
        if len(data[0][i]) < 7:
            continue
        if len(data[0][i]) > 30:
            continue
        ins = list([[s[0] for s in data[0][i]], [s[1] for s in data[0][i]], [s[2] for s in data[0][i]]])
        for j in range(1, len(data)):
            ins.append([s[2] for s in data[j][i]])
        predictions.append(ins)

    print(len(predictions))

    head = ['WORD', 'LA', 'GT', 'MV', 'LC', 'PG', 'MVS', 'LCS', 'PGS']
    for p in predictions:  # 422
        for r, h in zip(p, head):
            print(h, '\t', '\t'.join(r))
        input('任意键继续')

    # scores_by_worker()
    return


if __name__ == "__main__":
    main()
