# coding:utf-8
# @Time     : 2021/8/20 2:34 PM
# @Author   : jeffrey

import conllu
from conllu import parser
import argparse

def load_conll(f):
    data = []
    sents = f.read().strip().split("\n\n")
    for sent in sents:
        data.append([line.strip().split("\t") for line in sent.strip().split("\n") if not line.startswith("#")])
    return data


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='shell para')
    args_parser.add_argument('--file_path', type=str, help='conllu file')
    args = args_parser.parse_args()
    data = load_conll(open(args.file_path))
    print("conllu总得句子数目：%d"%len(data))
