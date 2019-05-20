import pyphen


def preprocess(DATA):
    data = DATA
    data = data.lower().replace('"', ''). \
            replace('<br> ', '\n').replace('<br>', '\n').replace('\r', '').replace('.', ''). \
            replace('?', '').replace('!', '').replace("&amp", ''). \
            replace('(', '').replace(')', '').replace('*', '').replace("'", ''). \
            replace('[', '').replace(']', '').replace('/', '').replace("\\", ''). \
            replace('ę', 'e').replace('ć', 'c').replace('ą', 'a'). \
            replace('ń', 'n').replace('ó', 'o').replace('ś', 's').replace('ź', 'z'). \
            replace('ż', 'z').replace('ł', 'l').replace(',', '').replace('\'', '')
    data = data.replace("--", "-").replace("-", " ")
    return data


def preprocess_pl(DATA):
    data = DATA
    data = data.replace('"', '').replace('\n\n', '\n'). \
        replace('<br> ', '\n').replace('<br>', '\n').replace('\r', '').replace("&amp", ''). \
        replace('(', '').replace(')', '').replace('*', '').replace("'", ''). \
        replace('[', '').replace(']', '').replace('/', '').replace("\\", ''). \
        replace(',', '').replace('\'', '').replace('—', '').replace("\ufeff", '')
    data = data.replace("--", "-").replace("-", " ")
    return data


def get_syllables(data, lang):

    data = data.lower().replace('.', '').replace('?', '').replace('!', '')
    preprocessed =[]
    data = data.replace("--", "-").replace("-", " ")
    dic = pyphen.Pyphen(lang=lang)
    for line in data.split('\n'):
        linebuf = []
        for word in line.split(' '):
            linebuf += dic.inserted(word).split('-') + [' ']
        preprocessed += linebuf + ['\n']

    return preprocessed
