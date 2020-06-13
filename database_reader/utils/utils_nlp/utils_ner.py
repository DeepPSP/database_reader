# -*- coding: utf-8 -*-
"""
docstring, to write
"""
import codecs
from typing import NoReturn


def bio_to_bioes(input_fp:str, output_fp:str, sep:str) -> NoReturn:
    """ not finished, not checked,

    convert the bio format annotated file to bioes format

    Parameters:
    -----------
    input_fp: str,
        the path of the input BIO format file
    output_fp: str,
        the path of the output BIOES format file
    sep: str,
        separator of the word and the label of each line in the input file
    """
    with codecs.open(input_fp, 'r', encoding='utf-8') as in_f, codecs.open(output_fp, 'w', encoding='utf-8') as out_f:
        words = []
        labels = []
        for line in in_f:
            contends = line.strip()
            tokens = contends.split(sep)
            if len(tokens) == 2:
                words.append(tokens[0])
                labels.append(tokens[1])
            else:
                if len(contends) == 0:
                    labels = [label for label in labels if len(label) > 0]
                    words = [word for word in words if len(word) > 0]
                    # convert to BIOES format
                    for idx, b in enumerate(labels):
                        if b.startswith("B") and (idx==len(labels)-1 or not labels[idx+1].startswith("I")):
                            labels[idx] = labels[idx].replace("B","S")
                        elif b.startswith("I") and (idx==len(labels)-1 or not labels[idx+1].startswith("I")):
                            labels[idx] = labels[idx].replace("I","E")
                    for w, b in zip(words, labels):
                        out_f.write(f"{w}{sep}{b}\n")
                    out_f.write("\n")
                    words = []
                    labels = []
                    continue
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
    
