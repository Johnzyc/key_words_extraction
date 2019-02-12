# -*- coding: utf-8 -*-
import os
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser
from typing import *
from collections import Counter


LTP_MODEL_DIR = "model_files/ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_MODEL_DIR, "cws.model")
pos_model_path = os.path.join(LTP_MODEL_DIR, "pos.model")
par_model_path = os.path.join(LTP_MODEL_DIR, 'parser.model')

custom_dict_path = "library/custom_dict.txt"
stop_words_path = "library/stop_words.txt"
degree_words_path = "library/degrees/"


class LtpTool(object):

    _instance = None

    def __init__(self):
        self.segmentor = Segmentor()
        self.postagger = Postagger()
        self.parser = Parser()
        self.stop_words = set()
        self.degree_words = dict()

    def load_model(self):
        self.segmentor.load_with_lexicon(cws_model_path, custom_dict_path)
        self.postagger.load(pos_model_path)
        self.parser.load(par_model_path)
        self._load_stop_words()
        self._load_degree_words()
        # print(self.stop_words)

    def _load_degree_words(self):
        with open(degree_words_path+"extreme.txt", 'r', encoding='utf-8') as extreme:
            for word in extreme.readlines():
                self.degree_words.setdefault("非常", set())
                self.degree_words["非常"].add(word)
        with open(degree_words_path+"insufficiently.txt", 'r', encoding='utf-8') as insufficiently:
            for word in insufficiently.readlines():
                self.degree_words.setdefault("不怎么", set())
                self.degree_words["不怎么"].add(word)
        with open(degree_words_path+"ish.txt", 'r', encoding='utf-8') as ish:
            for word in ish.readlines():
                self.degree_words.setdefault("有点儿", set())
                self.degree_words["有点儿"].add(word)
        with open(degree_words_path+"more.txt", 'r', encoding='utf-8') as more:
            for word in more.readlines():
                self.degree_words.setdefault("更加", set())
                self.degree_words["更加"].add(word)
        with open(degree_words_path+"over.txt", 'r', encoding='utf-8') as over:
            for word in over.readlines():
                self.degree_words.setdefault("过于", set())
                self.degree_words["过于"].add(word)
        with open(degree_words_path+"very.txt", 'r', encoding='utf-8') as very:
            for word in very.readlines():
                self.degree_words.setdefault("特别", set())
                self.degree_words["特别"].add(word)

    def _load_stop_words(self):
        with open(stop_words_path, 'r', encoding='utf-8') as stop:
            for word in stop.readlines():
                self.stop_words.add(word.strip())

    def words_segmentor(self, sentence: str):  #分词
        words = self.segmentor.segment(sentence)
        return [w for w in words if w not in self.stop_words]

    def words_postagor(self, words: List[str]):  #词性标注
        return self.postagger.postag(words)

    def parse_analysor(self, words: List[str], postags: List[str]):  #依存句法
        return self.parser.parse(words, postags)

    @staticmethod
    def get_ltp_tool():
        if not LtpTool._instance:
            ltp_tool = LtpTool()
            ltp_tool.load_model()
            LtpTool._instance = ltp_tool
        return LtpTool._instance


def generate_pair(pair_dict: dict):
    p_dict = dict()
    for key, value in pair_dict.items():
        p_dict.setdefault(value, [])
        p_dict[value].append(key)
    r_list = []
    for k, v in p_dict.items():
        v.append(k)
        r_list.append(v)
    return r_list


def parse_analysis(sentence: str):
    #
    ltp_tool = LtpTool.get_ltp_tool()
    words = ltp_tool.words_segmentor(sentence)
    postags = ltp_tool.words_postagor(words)
    arcs = ltp_tool.parse_analysor(words, postags)
    # print([x for x in zip(words, postags)])
    # print("\t".join("%s:%s" % (words[arc.head-1], arc.relation) for arc in arcs))
    sbv_pair, adv_pair, rad_pair = {}, {}, {}
    ban_speech = {'q', 'm'}
    for i in range(len(arcs)):
        obj_index = arcs[i].head-1
        if arcs[i].relation == 'SBV' or arcs[i].relation == 'ATT' and postags[obj_index] not in ban_speech:
            # print("post", words[obj_index], postags[obj_index])
            sbv_pair[i] = obj_index
        if arcs[i].relation == 'ADV':
            adv_pair[i] = obj_index
        if arcs[i].relation == 'RAD' or arcs[i].relation == 'VOB':
            rad_pair[obj_index] = i
    # for i in rad_pair:
    #     print("rad:", words[i], words[rad_pair[i]])
    new_words = []
    for i in range(len(words)):
        word = words[i]
        flag = postags[i]
        if flag in ban_speech:
            word = ""
        new_words.append(word)
    candidates = []
    for p_list in generate_pair(adv_pair):
        obj_i = p_list[-1]
        new_words[obj_i] = "".join([new_words[x] for x in sorted(p_list)])
        if 'a' == postags[obj_i]:
            candidates.append(new_words[obj_i])
        # print(new_words[obj_i])
    for key in rad_pair:
        obj_i = rad_pair[key]
        new_words[key] = new_words[key] + new_words[obj_i] if key <= obj_i else new_words[obj_i] + new_words[key]
        # print(new_words[key])
    for key in sbv_pair:
        if sbv_pair[key] == "":
            continue
        if key+1 in sbv_pair:
            order_i = sorted([key, sbv_pair[key], sbv_pair[key+1]])
            candidate = new_words[order_i[0]] + new_words[order_i[1]] + new_words[order_i[2]]
            if new_words[sbv_pair[key]] == new_words[sbv_pair[key+1]]:
                continue
            # print(new_words[key], new_words[sbv_pair[key]], new_words[sbv_pair[key+1]])
            sbv_pair[key+1] = ""
        else:
            obj_i = sbv_pair[key]
            candidate = new_words[key] + new_words[obj_i] if key <= obj_i else new_words[obj_i] + new_words[key]
            # print(candidate)
        candidates.append(candidate)
    return candidates


def generate_candidates(file_path: str):
    comment_dict = dict()
    candi_dict = dict()
    total_counter = Counter()
    total_record = dict()
    with open(file_path, "r", encoding="utf-8") as xmh:
        for comment in xmh.readlines():
            comment = comment.split(":")
            _id = comment[0]
            comment = "".join(comment[1:])
            comment_dict[_id] = comment
            sents = SentenceSplitter.split(comment.strip())
            candi_dict.setdefault(_id, [])
            for sent in sents:
                candidates = parse_analysis(sent)
                candi_dict.get(_id).extend(candidates)
            for key_words in candi_dict.get(_id):
                total_counter.setdefault(key_words, 0)
                total_counter[key_words] += 1
                total_record.setdefault(key_words, set())
                total_record[key_words].add(_id)
    # print(candi_dict)
    # print(total_record)
    # print(total_counter.most_common(100))
    return comment_dict, candi_dict, total_counter, total_record


if __name__ == "__main__":
    s = "看上去挺不错的，现在还是硬的，要放一放。包装也很好。"
    print(parse_analysis(s))
    # ltp_tool = LtpTool.get_ltp_tool()
    # words = ltp_tool.words_segmentor(s)
    # flags = ltp_tool.words_postagor(words)
    # print(list(zip(words, flags)))
    # xmh = "data_files/yit_comments_63162.txt"
    # com_dict, candidates, candi_counter, word2comment = generate_candidates(xmh)
    # word_counter = Counter()
    # for tag in candi_counter.keys():
    #     ltp_tool = LtpTool.get_ltp_tool()
    #     words = ltp_tool.words_segmentor(tag)
    #     postags = ltp_tool.words_postagor(words)
    #     for i in range(len(words)):
    #         word = words[i]
    #         flag = postags[i]
    #         if flag.startswith("n") or flag == "a" and len(word) > 1:
    #             word_counter.setdefault(word, 0)
    #             word_counter[word] += 1
    # print(word_counter.most_common(10))