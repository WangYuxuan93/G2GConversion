__author__ = 'max'

from neuronlp2.io.instance import Sentence
from neuronlp2.io.common import ROOT, ROOT_POS, ROOT_CHAR, ROOT_TYPE, END, END_POS, END_CHAR, END_TYPE,PAD_TYPE,PAD
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH
from neuronlp2.mappings.ud_mapping import ud_v2_en_label_mapping
import re

class G2GInstance(object):
    def __init__(self, sentence, postags, pos_ids, heads, types, type_ids, src_heads, src_types, src_type_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids

        self.src_heads = src_heads
        self.src_types = src_types
        self.src_type_ids = src_type_ids

    def length(self):
        return self.sentence.length()


class CoNLLUReaderG2G(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, pre_alphabet=None, pos_idx=4,
                 old_labels=None):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet
        self.__pre_alphabet = pre_alphabet
        self.__old_alphabet = old_labels
        self.pos_idx = pos_idx

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            if line.startswith('#'):
                line = self.__source_file.readline()
                continue
            items = line.split()
            if re.match('[0-9]+[-.][0-9]+', items[0]):
                line = self.__source_file.readline()
                continue
            lines.append(items)
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        src_types = []
        src_type_ids = []
        src_heads = []
        if self.__pre_alphabet:
            pres = []
            pre_ids = []
        else:
            pres = None
            pre_ids = None

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            types.append([]) # Jeffrey: heads and types should be a list
            type_ids.append([])
            heads.append([])
            # source graph
            src_types.append([])
            src_type_ids.append([])
            src_heads.append([])

            if self.__pre_alphabet:
                pres.append(ROOT)
                pre_ids.append(self.__pre_alphabet.get_index(ROOT))

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)
            word = tokens[1]
            pos = tokens[self.pos_idx]
            
            headlist = []
            typelist = []
            for x in tokens[8].split("|"):
                if x != '_':
                    p = x.split(":",1)
                    headlist.append(int(p[0]))
                    typelist.append(p[1])
            heads.append(headlist)
            types.append(typelist)
            #  exception:
            temp=[]
            for type in typelist:
                try:
                    temp_type = self.__type_alphabet.get_index(type)
                    temp.append(temp_type)
                except:
                    temp_type = self.__type_alphabet.get_index(PAD_TYPE)  # Jeffrey type不存在的情况
                    # temp_type = self.__type_alphabet.next_index
                    # self.__type_alphabet.next_index +=1
                    print("【ERROR arc_type:%s】"%type)
                    temp.append(temp_type)
            type_ids.append(temp)

            # source graph
            src_headlist = []
            src_typelist = []
            for x in tokens[9].split("|"):
                if x != '_':
                    p = x.split(":",1) #EMNLP论文中的原始代码并没有改过来，可能原因在于GAT没用利用弧上的标签信息
                    src_headlist.append(int(p[0]))
                    src_typelist.append(p[1])
            src_heads.append(src_headlist)
            src_types.append(src_typelist)
            src_temp=[]
            # for type in src_typelist:
            #     try:
            #         temp_type = ud_v2_en_label_mapping[type]
            #         src_temp.append(temp_type)
            #     except:
            #         temp_type = ud_v2_en_label_mapping["<PAD>"]
            #         # temp_type = self.__type_alphabet.next_index
            #         # self.__type_alphabet.next_index +=1
            #         print("【ERROR arc_type:%s】"%type)
            #         src_temp.append(temp_type)
            # src_type_ids.append(src_temp)
            # jeffrey 2021-9-12
            for type in src_typelist and self.__old_alphabet:
                try:
                    temp_type = self.__old_alphabet.get_index(type)
                    src_temp.append(temp_type)
                except:
                    temp_type = self.__old_alphabet.get_index(PAD_TYPE)  # Jeffrey type不存在的情况
                    # temp_type = self.__type_alphabet.next_index
                    # self.__type_alphabet.next_index +=1
                    print("【ERROR arc_type:%s】" % type)
                    src_temp.append(temp_type)

            # save original word in words (data['SRC']), to recover this for normalize_digits=True
            words.append(word)
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            if self.__pre_alphabet:
                pres.append(word)
                id = self.__pre_alphabet.get_index(word)
                if id == 0:
                    id = self.__pre_alphabet.get_index(word.lower())
                pre_ids.append(id)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append([])
            type_ids.append([])
            heads.append([])

            src_types.append([])
            src_type_ids.append([])
            src_heads.append([])
            if self.__pre_alphabet:
                pres.append(END)
                pre_ids.append(self.__pre_alphabet.get_index(END))

        return G2GInstance(Sentence(words, word_ids, char_seqs, char_id_seqs, pres=pres, pre_ids=pre_ids, lines=lines), 
                            postags, pos_ids, heads, types, type_ids, src_heads, src_types, src_type_ids)