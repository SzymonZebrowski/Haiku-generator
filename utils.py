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

    preprocessed =[]
    data = data.replace("--", "-").replace("-", " ")
    dic = pyphen.Pyphen(lang='en')
    for line in data.split('\n'):
        linebuf = []
        for word in line.split(' '):
            linebuf += dic.inserted(word).split('-') + [' ']
        preprocessed += linebuf + ['\n']


    return preprocessed