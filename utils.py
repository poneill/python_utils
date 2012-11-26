import random

def mean(xs):
    return sum(xs)/float(len(xs))

def median(xs):
    mp = int(len(xs)/2)
    if len(xs) % 2 == 1:
        return sorted(xs)[mp]
    else:
        return mean(sorted(xs)[mp-1:mp+1])
    
def variance(xs,correct=True):
    n = len(xs)
    correction = n/float(n-1) if correct else 1
    return correction * mean([x**2 for x in xs]) - mean(xs) ** 2    

def zipWith(f,xs,ys):
    return map(lambda(x,y):f(x,y),zip(xs,ys))

def transpose(xxs):
    return zip(*xxs)

def normalize(xs):
    total = float(sum(xs))
    return [x/total for x in xs]

def frequencies(xs):
    length = float(len(xs))
    return [xs.count(x)/length for x in set(xs)]

def verbose_gen(xs,modulus=1):
    for i,x in enumerate(xs):
        if not i % modulus:
            print i
        yield x
        
def h(ps):
    return -sum([p * log2(p) for p in ps])
print "loaded utils"

def safe_log2(x):
    """Implements log2, but defines log2(0) = 0"""
    return math.log(x,2) if x > 0 else 0

def group_by(xs,n):
    return [xs[i:i+n] for i in range(0,len(xs),n)]
    
def split_on(xs, pred):
    """Split xs into a list of lists each beginning with the next x
    satisfying pred, except possibly the first"""
    indices = [i for (i,v) in enumerate(xs) if pred(v)]
    return [xs[i:j] for (i,j) in zip([0]+indices,indices+[len(xs)]) if i != j]

def separate(pred, lst):
    """separates lst into a list of elements satisfying pred and a list of 
    elements not satisfying it.
    """
    sheep = []
    goats = []
    for elem in lst:
        if pred(elem):
            sheep.append(elem)
        else:
            goats.append(elem)
    return (sheep, goats)

def nmers(n):
    if n == 1:
        return ["A","C","G","T"]
    else:
        return sum([map(lambda(b):b+c,nmers(n-1)) for c in base_pair_ordering],[])

def complement(base):
    return {"A":"T","T":"A","G":"C","C":"G"}[base]
    
def wc(word):
    return "".join(map(complement, word[::-1]))

def pprint(x):
    for row in x:
        print row

def choose2(xs,gen=False):
    """return list of choose(xs, 2) pairs, retaining ordering on xs"""
    if gen:
        return ((x1, x2) for i, x1 in enumerate(xs) for x2 in xs[i+1:])
    else:
        return [(x1, x2) for i, x1 in enumerate(xs) for x2 in xs[i+1:]]

def pairs(xs):
    return zip(xs[:-1],xs[1:])

def partition(pred, xs):
    part = []
    appended = False
    for x in xs:
        appended = False
        for p in part:
            if pred(x,p[0]):
                p.append(x)
                appended = True
                break
        if not appended:
            part.append([x])
    return part

def foldl(f,z,xs):
    if not xs:
        return z
    else: 
        return foldl(f,f(z,xs[0]),xs[1:])

def foldl1(f,xs):
    return foldl(f,xs[0],xs[1:])

# naive implementation borrowed from stack overflow

def levenshtein(seq1, seq2):
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]

def sample(n,xs,replace=True):
    if replace:
        return [random.choice(xs) for i in range(n)]
    else:
        ys = xs[:]
        samp = []
        for i in range(n):
            y = random.choice(ys)
            samp.append(y)
            ys.remove(y)
        return samp

def matrix_mult(A,B):
    """Given two row-major matrices as nested lists, return the matrix
    product"""
    return [[sum(A[i][k] * B[k][j] for k in range(len(A[0])))
             for j in range(len(B[0]))]
            for i in range(len(A))
]
