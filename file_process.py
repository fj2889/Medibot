# =================================
# using for read write files
# =================================
import pickle
from prepare_data import Dataset # 声明定义域

def savefile(savepath, content):
    '''
    save files
    '''
    try:
        fp = open(savepath, 'w', encoding='utf8', errors='ignore')
        fp.write(content)
        fp.close()
    except Exception as e:
        print(e)


def readfile(path):
    '''
    read files
    '''
    try:
        fp = open(path, "r", encoding='utf8', errors='ignore')
        content = fp.read()
        fp.close()
    except Exception as e:
        print(e)

    return content


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        result= pickle.load(f)
        return result
