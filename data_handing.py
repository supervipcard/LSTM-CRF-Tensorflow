import json
from collections import Counter


def dict2json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        line = json.dumps(data, ensure_ascii=False)
        f.write(line)


def json2dict(file):
    with open(file, 'r', encoding='utf-8') as f:
        item = f.read()
        data = json.loads(item)
    return data


def set_vocabulary(data):
    words = ''.join([i[0] for i in data])
    words_list = list(Counter(words).most_common())
    words_dict = {'UNK': 0}
    for key, value in enumerate(words_list):
        words_dict[value[0]] = len(words_dict)
    return words_dict


def set_train_data(data):
    words_dict = json2dict(vocabulary_file_path)
    sign_dict = {'TB': 1, 'TI': 2, 'CB': 3, 'CI': 4, 'PB': 5, 'PI': 6, 'NB': 7, 'NI': 8, 'O': 9}
    s = []
    for sentence in data:
        a = [words_dict[word] for word in sentence[0]]
        b = [sign_dict[word] for word in sentence[1]]
        s.append([a, b])
    return s


if __name__ == '__main__':
    data_file_path = './data/data.json'
    vocabulary_file_path = './data/vocabulary.json'
    train_data_file_path = './data/train_data.json'
    dict2json(set_vocabulary(json2dict(data_file_path)), vocabulary_file_path)
    dict2json(set_train_data(json2dict(data_file_path)), train_data_file_path)
