"""
    This file takes a CoQA data file as input and generates the input files for training the pipeline model.
"""


import argparse
import json
import re
import time
import string

from nltk.tokenize import word_tokenize,sent_tokenize

if False:
    from pycorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP('http://localhost:9000')

UNK = 'unknown'


def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s


"""
def process(text):
    paragraph = nlp.annotate(text, properties={
                             'annotators': 'tokenize, ssplit',
                             'outputFormat': 'json',
                             'ssplit.newlineIsSentenceBreak': 'two'})

    output = {'word': [],
              # 'lemma': [],
              # 'pos': [],
              # 'ner': [],
              'offsets': []}

    for sent in paragraph['sentences']:
        for token in sent['tokens']:
            output['word'].append(_str(token['word']))
            # output['lemma'].append(_str(token['lemma']))
            # output['pos'].append(token['pos'])
            # output['ner'].append(token['ner'])
            output['offsets'].append((token['characterOffsetBegin'], token['characterOffsetEnd']))
    return output
"""

#corenlpを用いて単語分割していたのをword_tokenizeでの分割に変更
#高速化するが、精度は知らない
def process(text):
    output = {'word': [],
              # 'lemma': [],
              # 'pos': [],
              # 'ner': [],
              'offsets': []}
    text_split=word_tokenize(text)
    position=0
    for t in text_split:
        start=position+text[position:].find(t)
        end=start+len(t)
        output['word'].append(_str(t))
        output['offsets'].append((start, end))
        position=end

    return output


def get_str(output, lower=False):
    s = ' '.join(output['word'])
    return s.lower() if lower else s


def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def find_span(offsets, start, end):
    start_index = end_index = -1
    for i, offset in enumerate(offsets):
        if (start_index < 0) or (start >= offset[0]):
            start_index = i
        if (end_index < 0) and (end <= offset[1]):
            end_index = i
    return (start_index, end_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', '-d', type=str, required=True)
    parser.add_argument('--output_file1', '-o1', type=str, required=True, help='Json file for the DrQA model')
    parser.add_argument('--output_file2', '-o2', type=str, required=True, help='Files for the seq2seq model')
    args = parser.parse_args()

    with open(args.data_file, 'r') as f:
        dataset = json.load(f)

    f_src = open('{}-src.txt'.format(args.output_file2), 'w')
    f_tgt = open('{}-tgt.txt'.format(args.output_file2), 'w')
    data = []
    start_time = time.time()
    for i, datum in enumerate(dataset['data']):
        if i % 10 == 0:
            print('processing %d / %d (used_time = %.2fs)...' %
                  (i, len(dataset['data']), time.time() - start_time))
        context_str = datum['story']
        _datum = {'context': context_str,
                  'source': datum['source'],
                  'id': datum['id'],
                  'filename': datum['filename']}
        _datum['annotated_context'] = process(context_str)
        _datum['qas'] = []
        _datum['context'] += UNK
        _datum['annotated_context']['word'].append(UNK)
        _datum['annotated_context']['offsets'].append(
            (len(_datum['context']) - len(UNK), len(_datum['context'])))
        assert len(datum['questions']) == len(datum['answers'])

        additional_answers = {}
        if 'additional_answers' in datum:
            for k, answer in datum['additional_answers'].items():
                if len(answer) == len(datum['answers']):
                    for ex in answer:
                        idx = ex['turn_id']
                        if idx not in additional_answers:
                            additional_answers[idx] = []
                        additional_answers[idx].append(ex['input_text'])

        for question, answer in zip(datum['questions'], datum['answers']):
            assert question['turn_id'] == answer['turn_id']
            idx = question['turn_id']
            _qas = {'turn_id': idx,
                    'question': question['input_text'],
                    'answer': answer['input_text']}
            if idx in additional_answers:
                _qas['additional_answers'] = additional_answers[idx]

            _qas['annotated_question'] = process(question['input_text'])
            _qas['annotated_answer'] = process(answer['input_text'])
            _qas['answer_span_start'] = answer['span_start']
            _qas['answer_span_end'] = answer['span_end']

            start = answer['span_start']
            end = answer['span_end']
            chosen_text = _datum['context'][start: end].lower()
            while len(chosen_text) > 0 and chosen_text[0] in string.whitespace:
                chosen_text = chosen_text[1:]
                start += 1
            while len(chosen_text) > 0 and chosen_text[-1] in string.whitespace:
                chosen_text = chosen_text[:-1]
                end -= 1
            input_text = _qas['answer'].strip().lower()

            #spanの中に解答がある場合はそのまま解答をspanにする
            #ない場合にはspanをspanとして扱う。
            #find_span:文字と単語の変換
            #pipelineの場合はそもそもanswerではなく、spanを扱うべき？
            if input_text in chosen_text:
                i = chosen_text.find(input_text)
                _qas['answer_span'] = find_span(_datum['annotated_context']['offsets'],
                                                start + i, start + i + len(input_text))
            else:
                _qas['answer_span'] = find_span(_datum['annotated_context']['offsets'], start, end)
            _datum['qas'].append(_qas)

            s, e = _qas['answer_span']
            span_str = ' '.join(_datum['annotated_context']['word'][s: e + 1]).lower()
            f_src.write('{} || {}\n'.format(get_str(_qas['annotated_question'], lower=True), span_str))
            f_tgt.write('{}\n'.format(get_str(_qas['annotated_answer'], lower=True)))
        data.append(_datum)
    f_src.close()
    f_tgt.close()

    dataset['data'] = data
    with open(args.output_file1, 'w') as output_file:
        json.dump(dataset, output_file, sort_keys=True, indent=4)
