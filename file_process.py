# =================================
# using for read write files
# =================================
import pickle
import multiprocessing

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



def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def pool_map(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]



# def pool_map(_map, _data):
#     '''
#     多线程map
#     '''
#
#     def parmap(f, X):
#         pipe = [Pipe() for x in X]
#         proc = [Process(target=spawn(f), args=(c, x)) for x, (p, c) in izip(X, pipe)]
#         [p.start() for p in proc]
#         [p.join() for p in proc]
#         return [p.recv() for (p, c) in pipe]
#
#
#     with mp.Pool(processes=(mp.cpu_count() - 1)) as pool:
#         def work(_data):
#             _data.work()
#
#         pool.apply_async(work, args=(foo,))
#         return pool.map(_map, _data)