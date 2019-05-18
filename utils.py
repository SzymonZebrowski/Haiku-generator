
def preprocess(DATA):
    data = DATA
    data = data.replace('"', ''). \
            replace('<br> ', '\n').replace('<br>', '\n').replace('\r', '').replace("&amp", ''). \
            replace('(', '').replace(')', '').replace('*', ''). \
            replace('[', '').replace(']', '').replace('/', '').replace("\\", ''). \
            replace('ę', 'e').replace('ć', 'c').replace('ą', 'a'). \
            replace('ń', 'n').replace('ó', 'o').replace('ś', 's').replace('ź', 'z'). \
            replace('ż', 'z').replace('ł', 'l')

    data = data.replace("--", "-").replace("-", " ")
    return data