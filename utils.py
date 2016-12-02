import random
from math import sqrt,log,exp,pi,sin,cos,gamma,acos,sqrt
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
import bisect
import sys
import numpy as np
import itertools
from scipy.special import gammaln
from scipy.stats import mannwhitneyu
from tqdm import *
from scipy.stats import pearsonr, spearmanr
epsilon = 10**-100

def translate_spins(spins):
    return "".join("ACGT"[i] for i in spins)
    
def iota(x):
    """Identity function.  Surprisingly useful to have around"""
    return x

def log2(x):
    return log(x,2)

def mean(xs):
    if hasattr(xs,"__len__"):
        return sum(xs)/float(len(xs))
    else:
        acc = 0
        n = 0
        for x in xs:
            acc += x
            n += 1
        return acc/float(n)

def mean_imp(xs):
    acc = 0
    n = 0
    for x in xs:
        acc += x
        n += 1
    return acc/float(n)

def geo_mean(xs):
    return product(xs)**(1.0/len(xs))

def median(xs):
    mp = int(len(xs)/2)
    if len(xs) % 2 == 1:
        return sorted(xs)[mp]
    else:
        return mean(sorted(xs)[mp-1:mp+1])

def mean_and_sd(xs):
    return mean(xs),sd(xs)

def mode(xs):
    counts = Counter(xs)
    x,count = max(counts.items(),key=lambda(x,y):y)
    return x

def variance(xs,correct=True):
    n = len(xs)
    correction = n/float(n-1) if correct else 1
    mu = mean(xs)
    return correction * mean([(x-mu)**2 for x in xs])

def sd(xs,correct=True):
    return sqrt(variance(xs,correct=correct))

def se(xs,correct=True):
    return sd(xs,correct)/sqrt(len(xs))

def mean_ci(xs):
    """Return 95% CI for mean"""
    mu = mean(xs)
    s = 1.96 * se(xs)
    return (mu - s,mu + s)
    
def coev(xs,correct=True):
    return sd(xs,correct)/mean(xs)

def find_index(p,xs):
    """From Haskell Data.List.  find index of first element satisfying predicate, or None if no element."""
    for i,x in enumerate(xs):
        if p(x):
            return i
    return None

def zipWith(f,xs,ys):
    return map(lambda(x,y):f(x,y),zip(xs,ys))

def transpose(xxs):
    return zip(*xxs)

def normalize(xs):
    total = float(sum(xs))
    return [x/total for x in xs]

def log_sum(log_xs):
    "given log_xs, return log(sum(xs))"
    log_xmax = max(log_xs)
    return log_xmax + log(sum(exp(log_x - log_xmax) for log_x in log_xs))

def np_log_sum(log_xs):
    "given numpy array log_xs, return log(sum(xs))"
    log_xmax = np.max(log_xs)
    return log_xmax + log(np.sum(np.exp(log_xs - log_xmax)))
    #return log_xmax + log(sum(exp(log_x - log_xmax) for log_x in log_xs))
    
def log_normalize(log_xs):
    """return log_xs' such that sum(xs') == 1'"""
    log_Z = log_sum(log_xs)
    return [log_x - log_Z for log_x in log_xs]

def np_log_normalize(log_xs):
    log_Z = np_log_sum(log_xs)
    return log_xs - log_Z
    
def frequencies(xs):
    # faster than either of frequencies_ref [!]
    length = float(len(xs))
    return [xs.count(x)/length for x in set(xs)]

def frequencies_ref(xs):
    length = 0
    counts = {}
    for x in xs:
        if not x in counts:
            counts[x] = 1
        else:
            counts[x] += 1
        length += 1
    length = float(length)
    return [count/length for count in counts.values()]

def frequencies_ref2(xs):
    n = float(len(xs))
    return [v/n for v in Counter(xs).values()]
    
def unique(xs):
    us = []
    for x in xs:
        if not x in us:
            us.append(x)
    return us

def verbose_gen(xs,modulus=1):
    for i,x in enumerate(xs):
        if not i % modulus:
            print "%s\r" % i,
            sys.stdout.flush()
        yield x
        
def h(ps):
    """compute entropy (in bits) of a probability distribution ps"""
    return -sum([p * safe_log2(p) for p in ps])

def entropy(xs,correct=True,A=None):
    """compute entropy (in bits) of a sample from a categorical
    probability distribution"""
    if A == None:
        A = len(set(xs)) # NB: assuming every element appears!
    ps = frequencies(xs)
    correction = ((A - 1)/(2*log(2)*len(xs)) if correct
                  else 0) #Basharin 1959
    #print "correction:",correction
    return h(ps) + correction

def dna_entropy(xs,correct=True):
    """compute entropy (in bits) of a DNA sample"""
    ps = frequencies(xs)
    correction = (3)/(2*log(2)*len(xs)) #Basharin 1959
    #print "correction:",correction
    return h(ps) + (correction if correct else 0)

def motif_entropy(motif,correct=True,A=4):
    """Return the entropy of a motif, assuming independence"""
    return sum(map(lambda col:entropy(col,correct=correct,A=A),
                   transpose(motif)))

def columnwise_ic(motif,correct=True):
    return map(lambda col:2-dna_entropy(col,correct=correct),
                   transpose(motif))

def motif_ic(motif, correct=True, A=4):
    """Return the entropy of a motif, assuming independence and a
    uniform genomic background"""
    L = len(motif[0])
    return (log2(A) * L - motif_entropy(motif, correct=correct, A=A))

def mi(xs,ys,correct=True):
    """Compute mutual information (in bits) of samples from two
    categorical probability distributions"""
    hx  = entropy(xs,correct=correct)
    hy  = entropy(ys,correct=correct)
    hxy = entropy(zip(xs,ys),correct=correct)
    return hx + hy - hxy

def dna_mi_dep(xs,ys):
    hx  = entropy(xs,correct=True, alphabet_size=4)
    hy  = entropy(ys,correct=True, alphabet_size=4)
    hxy = entropy(zip(xs,ys),correct=True, alphabet_size=16)
    return hx + hy - hxy
    
def dna_mi(xs,ys):
    N = float(len(xs))
    joint = defaultdict(int,{k:v/N for k,v in Counter(zip(xs,ys)).items()})
    margx = {bx:sum(joint[(bx,by)] for by in "ACGT") for bx in "ACGT"}
    margy = {by:sum(joint[(bx,by)] for bx in "ACGT") for by in "ACGT"}
    return sum(joint[x,y]*log2(joint[x,y]/(margx[x]*margy[y])) if joint[x,y] else 0
               for x in "ACGT" for y in "ACGT")

def motif_mi(motif):
    """compute pairwise motif mi without any sample size correction"""
    cols = transpose(motif)
    return sum(dna_mi(col1, col2) for col1, col2 in choose2(cols))
    
def mi_table(xs,ys,display=False,normalize=False,f=iota):
    x_vals = sorted(set(xs))
    y_vals = sorted(set(ys))
    N = float(len(xs))
    assert len(ys) == N
    x_freqs = {k:v/N for (k,v) in Counter(xs).items()}
    y_freqs = {k:v/N for (k,v) in Counter(ys).items()}
    joint_freqs = Counter({k:v/N for (k,v) in Counter(zip(xs,ys)).items()})
    def denom(x,y):
        if normalize:
            return (x_freqs[x]*y_freqs[y])
        else:
            return 1
    table = [[f(joint_freqs[x,y]/denom(x,y)) for y in y_vals] for x in x_vals]
    if display:
        print " ".join(y_vals)
        for i,x in enumerate(x_vals):
            print x,
            for j,y in enumerate(y_vals):
                print table[i][j],
            print
    return table
    
# def dna_mi(xs,ys):
#     """Compute mutual information (in bits) of samples from two
#     nucleotide distributions, correcting for undersampling in entropy
#     calculation."""
#     hx = dna_entropy(xs)
#     hy = dna_entropy(ys)
#     hxy = entropy(zip(xs,ys),correct=True,alphabet_size=16)
#     return hx + hy - hxy

def dna_mi2(xs,ys):
    """Compute mutual information (in bits) of samples from two
    nucleotide distributions, correcting for undersampling in entropy
    calculation."""
    hx = entropy(xs,correct=False)
    hy = entropy(ys,correct=False)
    hxy = entropy(zip(xs,ys),correct=False)
    k = 16
    n = len(xs)
    expected_bias = (k-1)**2/float(2*n)
    return hx + hy - hxy - expected_bias

def permute(xs):
    """Return a random permutation of xs"""
    #return np.random.permutation(xs)
    xs_ = list(xs[:])
    random.shuffle(xs_)
    return xs_

def test_permute(xs):
    permutation = permute(xs)
    return all(permutation.count(x) == xs.count(x) for x in set(xs))

def mi_permute(xs,ys,n=100,conf_int=False,p_value=False,zero=False,mi_method=mi):
    """For samples xs and ys, compute an expected MI value (or
    confidence interval, a p_value for obtaining MI at least as high)
    by computing the MI of randomly permuted columns xs and ys n
    times, or MI(xs,ys) - the mean MI of the replicates"""
    replicates = [mi_method(permute(xs),permute(ys)) for i in range(n)]
    if conf_int:
        replicates = sorted(replicates)
        lower,upper = (replicates[int(n*.05)],replicates[int(n*.95)])
        assert lower <= upper
        return (lower,upper)
    elif p_value:
        replicates = sorted(replicates)
        mi_obs = mi_method(xs,ys)
        return len(filter(lambda s: s >= mi_obs,replicates))/float(n)
    elif zero:
        mi_obs = mi_method(xs,ys)
        return mi_obs - mean(replicates)
    else:
        return mean(replicates)

def safe_log2(x):
    """Implements log2, but defines log2(0) = 0"""
    return log(x,2) if x > 0 else 0

def round_up(x):
    return int(x) + (x%1 > 0)

def group_by(xs,n):
    chunks = [xs[i:i+n] for i in range(0,len(xs),n)]
    return chunks

def group_into(xs,n):
    """Group xs into n chunks, without preserving order"""
    chunks = [[] for __ in range(n)]
    for i,x in enumerate(xs):
        chunks[i%n].append(x)
    return chunks
    
def split_on(pred, xs):
    """Split xs into a list of lists each beginning with the next x
    satisfying pred, except possibly the first"""
    indices = [i for (i,v) in enumerate(xs) if pred(v)]
    return [xs[i:j] for (i,j) in zip([0]+indices,indices+[len(xs)]) if i != j]

def partition_according_to(f,xs):
    """Partition xs according to f"""
    part = []
    xsys = [(x,f(x)) for x in xs]
    yvals = unique([y for (x,y) in xsys])
    return [[x for (x,y) in xsys if y == yval] for yval in yvals]
    
    
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
    
def wc_ref(word):
    """Reference implementation of reverse complement method.  Not for
    use in production because Python is terrible"""
    return "".join(map(complement, word[::-1]))

def wc(word):
    """Reverse complement function"""
    # see wc_ref for non-terrible implementation
    new_word = ""
    for c in word[::-1]:
        new_word += {"A":"T","T":"A","G":"C","C":"G","N":"N"}[c] #~3x speedup by inlining
    return new_word

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

def fac(n):
    return gamma(n+1)

def log_fac_dep(n):
    """compute log n! for n <= 170, else approximate with stirling's"""
    if n <= 170:
        return log(fac(n))
    else:
        return (n + 1/2.0) * log(n) - n  + 1/2.0*log(2*pi)

def log_fac(n):
    return gammaln(n+1)
        
def choose(n,k):
    return fac(n)/(fac(k) * fac(n-k)) if n >= k else 0

def log_choose(n,k):
    return log_fac(n) - (log_fac(k) + log_fac(n-k))
    
def choose_reference(n,k):
    return prod((n-(k-i))/float(i) for i in range(1,k+1))

def concat(xxs):
    return [x for xs in xxs for x in xs]

def mmap(f,xxs):
    return [map(f,xs) for xs in xxs]

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
        N = len(xs)
        if n > N:
            raise Exception("n exceeds size of xs in sampling without replacement")
        samp = []
        js = []
        while len(js) < n:
            j = random.randrange(N)
            if not j in js:
                js.append(j)
        return [xs[j] for j in js]
    
def bs(xs):
    return sample(len(xs),xs,replace=True)

def bs_ci(f,xs,alpha=0.05,N=1000):
    fs = sorted([f(bs(xs)) for i in xrange(N)])
    i = int(alpha/2 * N)
    j = int(1 - alpha/2 * N)
    return fs[i],fs[j]
    
def fast_sample(n,xs):
    """Sample without replacement for large xs"""
    samp = []
    while len(samp) < n:
        x = random.choice(xs)
        if not x in samp:
            samp.append(x)
    return samp

def matrix_mult(A,B):
    """Given two row-major matrices as nested lists, return the matrix
    product"""
    return [[sum(A[i][k] * B[k][j] for k in range(len(A[0])))
             for j in range(len(B[0]))]
            for i in (range(len(A)))]

def identity_matrix(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]

def matrix_add(A,B):
    return [[A[i][j] + B[i][j] for j in range(len(B[0]))]
            for i in range(len(A))]

def matrix_scalar_mult(c,A):
    return mmap(lambda x:x*c,A)

def matrix_power(A,n):
    if n == 1:
        return A
    elif n % 2 == 0:
        return matrix_power(matrix_mult(A,A),n/2)
    else:
        return matrix_mult(A,matrix_power(A,n-1))

def boolean_matrix_mult(A,B):
    """Given two row-major boolean matrices as nested lists, return
    the matrix product"""
    print "in boolean matrix mult"
    return [[any(A[i][k] * B[k][j] for k in xrange(len(A[0])))
             for j in xrange(len(B[0]))]
            for i in verbose_gen(xrange(len(A)))]

def iterate_ref(f,x,n):
    if n == 0:
        return x
    else:
        return iterate(f,f(x),n-1)

def iterate(f,x,n):
    for i in xrange(n):
        x = f(x)
    return x
    
def iterate_list_ref(f,x,n):
    if n == 0:
        return [x]
    else:
        return  [x] + iterate_list(f,f(x),n-1)

def iterate_list(f,x,n):
    xs = [x]
    for i in xrange(n):
        x = f(x)
        xs.append(x)
    return xs
    
def converge(f,x,verbose=False,i=0):
    if verbose:
        print i
    y = f(x)
    if y == x:
        return x
    else:
        return converge(f,y,verbose=verbose,i=i+1)

def converge2(f,x,verbose=False,i=0):
    y = f(x)
    while not y == x:
        if verbose:
            print i
            i += 1
        x = y
        y = f(x)
    return y

def converge_list(f,x):
    y = f(x)
    history = [x,y]
    while not y == x:
        x = y
        y = f(x)
        history.append(y)
    return history

def data2csv(data, filename, sep=", ",header=None,overwrite=False):
    import os
    make_line = lambda row: sep.join([str(field) for field in row]) + "\n"
    if filename in os.listdir('.') and not overwrite:
        print "found ",filename
        pass
    with open(filename, 'w') as f:
        if header:
            f.write(make_line(header))
        f.write("".join([make_line(row) for row in data]))

def dot_reference(u,v):
    """This implementation is apparently too cool for python.  see dot"""
    return sum(zipWith(lambda x,y:x*y,u,v))

def dot(xs,ys):
    """Fast dot product operation"""
    acc = 0
    for (x,y) in zip(xs,ys):
        acc += x * y
    return acc

def norm(u):
    return sqrt(dot(u,u))

def cosine_distance(u,v):
    return dot(u,v)/(norm(u)*norm(v))

def l1(xs,ys):
    return sum(zipWith(lambda x,y:abs(x-y),xs,ys))

def l2(xs,ys):
    return sqrt(sum(zipWith(lambda x,y:(x-y)**2,xs,ys)))

def linf(xs,ys):
    return max(zipWith(lambda x,y:abs(x-y),xs,ys))

def takeWhile(p,xs):
    if not xs or not p(xs[0]):
        return []
    else:
        return [xs[0]] + takeWhile(p,xs[1:])

def distance(xs,ys):
    return l2(xs,ys)

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
def bisect_interval_ref(f,xmin,xmax,ymin=None,ymax=None,tolerance=1e-10,verbose=False):
    if verbose:
        print xmin,xmax,ymin,ymax
    if ymin is None:
        ymin = f(xmin)
    if ymax is None:
        ymax = f(xmax)
    assert(sign(ymin)!= sign(ymax)), "ymin=%s,ymax=%s" % (ymin,ymax)
    x = (xmin + xmax)/2.0
    y = f(x)
    if abs(y) < tolerance:
        return x
    else:
        if sign(y) == sign(ymin):
            return bisect_interval(f,x,xmax,ymin=y,ymax=ymax,
                                   tolerance=tolerance,verbose=verbose)
        else:
            return bisect_interval(f,xmin,x,ymin=ymin,ymax=y,
                                   tolerance=tolerance,verbose=verbose)

def bisect_interval(f,xmin,xmax,ymin=None,ymax=None,xtol=1e-10, ytol=1e-10,
                    verbose=False):
    if ymin is None:
        ymin = f(xmin)
    if ymax is None:
        ymax = f(xmax)
    assert(sign(ymin)!= sign(ymax)), "ymin=%s,ymax=%s" % (ymin,ymax)
    x = (xmin + xmax)/2.0
    y = f(x)
    while (xmax - xmin) > xtol and abs(y) > ytol:
        if verbose:
            print xmin,xmax,ymin,ymax
        x = (xmin + xmax)/2.0
        y = f(x)
        if sign(y) == sign(ymin):
            xmin = x
            ymin = y
        else:
            xmax = x
            ymax = y
    return x
            
def bisect_interval_noisy_ref(f,x0,
                              tolerance=0.01,verbose=False,
                              lb=None,ub=None,w=1.0,iterations=None):
    """solve f(x) == 0 via Robbins-Monro algorithm"""
    xs = [x0]
    def a(i):
        return w/(i+1)
    iterator = itertools.count() if iterations is None else xrange(iterations)
    for i in iterator:
        xn = xs[-1]
        fxn = f(xn)
        xnp1 = -a(i)*fxn + xn
        if lb and xnp1 < lb:
            print "warning: hit lower bound"
            xnp1 = lb
        if ub and xnp1 > ub:
            print "warning: hit upper bound"
            xnp1 = ub
        xs.append(xnp1)
        if verbose:
            print i, xn, fxn, abs(xnp1 - xn)
        if iterations is None and i > 3:
            if abs(xs[-1] - xs[-4]) < tolerance:
                return xs[-1]
    return xs[-1]

def bisect_interval_noisy_spec(f,x0,
                              tolerance=0.01,verbose=False,
                              lb=None,ub=None,w=1.0,iterations=None):
    """solve f(x) == 0 via Robbins-Monro algorithm"""
    xs = [x0]
    ys = []
    def a(i):
        return w/(i+1)
    iterator = itertools.count() if iterations is None else xrange(iterations)
    for i in iterator:
        xn = xs[-1]
        yn = f(xn)
        ys.append(yn)
        xnp1 = -a(i)*yn + xn
        if lb and xnp1 < lb:
            print "warning: hit lower bound"
            xnp1 = lb
        if ub and xnp1 > ub:
            print "warning: hit upper bound"
            xnp1 = ub
        xs.append(xnp1)
        if verbose:
            print i, xnp1, xn, a(i), abs(xnp1 - xn)
        if iterations is None and i > 3:
            if abs(xs[-1] - xs[-4]) < tolerance:
                return xs[-1]
    return xs,ys

def bisect_interval_noisy(f,x0,iterations=None,xtol=0.01,verbose=False,lb=None,ub=None,w=1):
    """solve f(x) == 0 via Polyak-Juditsky algorithm"""
    xs = [x0]
    ys = []
    running_mean = x0
    def b(i):
        return w*(i)**(-1/2.0)
    iterator = (itertools.count(start=1) if iterations is None
                else xrange(1,iterations+1))
    for i in iterator:
        xn = xs[-1]
        yn = f(xn)
        xnp1 = -b(i)*yn + xn
        if lb is not None and xnp1 < lb:
            print "warning: hit lower bound"
            xnp1 = lb
        if ub is not None and xnp1 > ub:
            print "warning: hit upper bound"
            xnp1 = ub
        xs.append(xnp1)
        ys.append(yn)
        old_running_mean = running_mean
        running_mean = (old_running_mean*i+xnp1)/(i+1.0)
        if verbose:
            print i, running_mean, xnp1, xn, abs(running_mean - old_running_mean), mean(ys)
        if iterations is None and i > 3:
            if abs(running_mean - old_running_mean) < xtol:
                return mean(xs)
    return mean(xs)

def bisect_interval_home_rolled(f, x0,  iterations, a=1, lb=None,ub=None,
                                verbose=False):
    xs = [x0]
    def restrict(x):
        print "restricting:",x
        if not lb is None and x < lb:
            return lb
        elif not ub is None and x > ub:
            return ub
        else:
            return x
    for n in xrange(iterations):
        x = mean(xs)
        print "x:",x
        y = f(x)
        print "y:",y
        x_new = restrict(x - a*y)
        print "x_new:",x_new
        xs.append(x_new)
        print x, x_new, y
    return mean(xs)
        
def bisect_interval_noisy_kde(f, lb, ub, trials=100, iterations_per_trial=1):
    xs = np.linspace(lb, ub, trials)
    ys = map(lambda x:mean(f(show(x)) for _ in range(iterations_per_trial)), xs)
    fhat = kde_regress(xs,ys,monotonic=True)
    return bisect_interval(fhat,lb, ub)

def bisect_interval_kw(f,lb,ub, iterations=10, verbose=False):
    """find root of f in [lb, ub] via Kiefer-Wolfowitz algorithm"""
    g = lambda x:-f(x)**2
    x = (lb + ub)/2.0
    def a(n):
        return 1.0/(n)
    def c(n):
        return n**(-1/3.0)
    for n in range(1,iterations+1):
        x = x + a(n)*(g(x+c(n)) - g(x-c(n)))/c(n)
        if x < lb + c(n):
            x = lb + c(n)
        elif x > ub - c(n):
            x = ub - c(n)
        if verbose:
            print n, x
    return x
    
def secant_interval_ref(f,xmin,xmax,ymin=None,ymax=None,tolerance=1e-10,verbose=False):
    if verbose:
        print xmin,xmax,ymin,ymax
    if ymin is None:
        ymin = f(xmin)
    if ymax is None:
        ymax = f(xmax)
    assert(sign(ymin)!= sign(ymax)), "ymin=%s,ymax=%s" % (ymin,ymax)
    m = (ymax - ymin)/(xmax - xmin)
    x = xmax - ymax/m
    y = f(x)
    if abs(y) < tolerance:
        return x
    else:
        if sign(y) == sign(ymin):
            return secant_interval(f,x,xmax,ymin=y,ymax=ymax,tolerance=tolerance,verbose=verbose)
        else:
            return secant_interval(f,xmin,x,ymin=ymin,ymax=y,tolerance=tolerance,verbose=verbose)

def secant_interval(f,xmin,xmax,ymin=None,ymax=None,tolerance=1e-10,verbose=False):
    if ymin is None:
        ymin = f(xmin)
    if ymax is None:
        ymax = f(xmax)
    for iteration in xrange(1000):
        if verbose:
            print xmin,xmax,ymin,ymax
        assert(sign(ymin)!= sign(ymax)), "ymin=%s,ymax=%s" % (ymin,ymax)
        m = (ymax - ymin)/float(xmax - xmin)
        x = xmax - ymax/m
        y = f(x)
        if abs(y) < tolerance:
            return x
        else:
            if sign(y) == sign(ymin):
                xmin = x
                ymin = y
            else:
                xmax = x
                ymax = y

        
def secant_interval_robust(f,xmin,xmax,ymin=None,ymax=None,tolerance=1e-10,p=0.1):
    #print xmin,xmax,ymin,ymax
    if ymin is None:
        ymin = f(xmin)
    if ymax is None:
        ymax = f(xmax)
    assert(sign(ymin)!= sign(ymax)), "ymin=%s,ymax=%s" % (ymin,ymax)
    if random.random() > p:
        m = (ymax - ymin)/(xmax - xmin)
        x = xmax - ymax/m
    else:
        x = (xmax + xmin)/2.0
    y = f(x)
    if abs(y) < tolerance:
        return x
    else:
        if sign(y) == sign(ymin):
            return secant_interval_robust(f,x,xmax,ymin=y,ymax=ymax,tolerance=tolerance,p=p)
        else:
            return secant_interval_robust(f,xmin,x,ymin=ymin,ymax=y,tolerance=tolerance,p=p)

def percentile(x,xs):
    """Compute what percentile value x is in xs"""
    return len(filter(lambda y:y < x,xs))/float(len(xs))

def normal_model(xs):
    mu = mean(xs)
    sigma = sqrt(variance(xs))
    return [random.gauss(mu,sigma) for x in xs]

def solve_quadratic(a,b,c):
    discriminant = b**2-4*a*c
    return ((-b + sqrt(discriminant))/(2*a),(-b - sqrt(discriminant))/(2*a))

def show(x):
    print x
    return x

def myrange(start,stop,step):
    return map(lambda x: start + x*step,
               range(int((stop-start)/step)))

def grad_descent(f,x,y,ep_x=0.0001,ep_y=0.0001):
    "minimize f"
    z = f(x,y)
    best_z = None
    epsilon_shrinkage = 1
    while best_z is None or z <= best_z or True:
        z = best_z 
        choices = [(x + ep_x,y),(x - ep_x,y),(x,y + ep_y),(x,y - ep_y)]
        z_choices = map(lambda (x,y):f(x,y),choices)
        choice = min(zip(choices,z_choices),key=lambda(z,z_ch):z_ch)
        (x,y),best_z = choice
        ep_x *= epsilon_shrinkage
        ep_y *= epsilon_shrinkage
        print x,y,log(best_z),ep_x,ep_y
    return x,y

def find_connected_components(M):
    """Given a graph represented as a row-major adjacency matrix,
    return a list of M's components"""
    components = []
    n = len(M)
    while len(components) < n:
        found_yet = concat(components)
        i = min([i for i in range(n) if not i in found_yet])
    v = [[i] + [0] * (n - 1)]
    v_inf = converge(lambda x:boolean_matrix_mult(x,M),verbose=True)
    component = [i for i in range(n) if v_inf[i]]
    components.append(component)
    
def hamming(xs,ys):
    return sum(zipWith(lambda x,y:x!=y,xs,ys))

def fdr(ps,alpha=0.05):
    """Given a list of p-values and a desired significance alpha, find the
    adjusted q-value such that a p-value less than q is expected to be
    significant at alpha, via the Benjamini-Hochberg method
    (appropriate for independent tests).
    """
    ps = sorted(ps)
    m = len(ps)
    ks = [k for k in range(m) if ps[k]<= (k+1)/float(m)*alpha] #k+1 because pvals are 1-indexed.
    K = max(ks) if ks else None
    return ps[K] if K else None #if none are significant

def bhy(ps,alpha=0.05):
    """Given a list of p-values of arbitrarily correlated tests and a
    desired significance alpha, find the adjusted q-value such that a
    p-value less than q is expected to be significant at alpha, via
    the Benjamini-Yekutieli method.
    """
    ps = sorted(ps)
    m = len(ps)
    def c(m):
        return sum(1.0/i for i in range(1,m+1))
    ks = [k for k in range(m) if ps[k]<= (k+1)/(float(m)*c(m))*alpha] #k+1 because pvals are 1-indexed.
    K = max(ks) if ks else None
    return ps[K] if K else None #if none are significant

def bhy(ps,alpha=0.05):
    """Compute FDR using Benjamini-Hochberg-Yekutieli procedure"""
    ps = sorted(ps)
    m = len(ps)
    def c(n):
        return sum(1/float(i) for i in range(1,n+1))
    cm = c(m)
    ks = [k for k in range(m) if ps[k]<= k/(float(m)*cm)*alpha]
    K = max(ks) if ks else None
    return ps[K] if K else None #if none are significant
    
def hamming(xs,ys):
    return sum(zipWith(lambda x,y:x!=y,xs,ys))

def enumerate_mutant_neighbors(site):
    sites = [] # changed this to exclude site itself Tue Mar  5 13:52:48 EST 2013
    site = list(site)
    for pos in range(len(site)):
        old_base = site[pos]
        for base in "ATCG":
            if not base == old_base:
                site[pos] = base
                sites.append("".join(site))
        site[pos] = old_base
    return sites

def enumerate_mutant_sites(site,k=1):
    site_dict = {site:0}
    j = 0
    for j in range(k):
        print j
        d = site_dict.copy()
        for s in d:
            if site_dict[s] != j:
                continue
            neighbors = enumerate_mutant_neighbors(s)
            for neighbor in neighbors:
                if not neighbor in site_dict:
                    site_dict[neighbor] = j + 1
    return site_dict.keys()

def regexp_from_sites(sites):
    """Return a minimal regexp matching given sites"""
    pass        

def sorted_indices(xs):
    """Return a list of indices that puts xs in sorted order.
    E.G.: sorted_indices([40,10,30,20]) => [1,3,2,0]"""
    return [i for (i,v) in sorted(enumerate(xs),key=lambda(i,v):v)]

def indices_where(xs,p):
    return [i for (i,x) in enumerate(xs) if p(x)]

def test_sorted_indices(xs):
    si = sorted_indices(xs)
    return sorted(xs) == [xs[i] for i in si]

def total_motif_mi(motif):
    cols = transpose(motif)
    return sum([mi(col1,col2) for (col1,col2) in choose2(cols)])

def motif_mi_pp(motif):
    """pairwise mi per pair"""
    L = len(motif[0])
    return total_motif_mi(motif)/choose(L,2)

def random_site(n):
    return "".join(random.choice("ACGT") for i in range(n))

def random_motif(length,num_sites):
    return [random_site(length) for i in range(num_sites)]

def mutate_site(site):
    i = random.randrange(len(site))
    b = site[i]
    new_b = random.choice([c for c in "ACGT" if not c == b])
    return subst(site,new_b,i)

def mutate_motif(motif):
    i = random.randrange(len(motif))
    return [site if j!=i else mutate_site(site)
            for j,site in enumerate(motif)]

def mutate_motif_p(motif,p):
    motif_ = motif[:]
    n = len(motif)
    L = len(motif[0])
    N = n * L
    r = np.random.binomial(N,p)
    for _ in xrange(r):
        i = random.randrange(n)
        j = random.randrange(L)
        b = motif[i][j]
        new_b = random.choice([c for c in "ACGT" if not c == b])
        motif_[i] = subst(motif_[i],new_b,j)
    return motif_

def mutate_motif_p_exact(motif,p):
    motif_ = motif[:]
    n = len(motif)
    L = len(motif[0])
    N = n * L
    r = np.random.binomial(N,p)
    r_so_far = 0
    choices = []
    while r_so_far < r:
        i = random.randrange(n)
        j = random.randrange(L)
        if (i,j) in choices:
            continue
        else:
            choices.append((i,j))
            r_so_far += 1
            b = motif[i][j]
            new_b = random.choice([c for c in "ACGT" if not c == b])
            motif_[i] = subst(motif_[i],new_b,j)
    return motif_

def subst_motif(motif,i,j,new_b):
    motif_ = motif[:]
    motif_[i] = subst(motif[i],new_b,j)
    return motif_
    
def get_ecoli_genome(at_lab=True):
    lab_file = "/home/poneill/ecoli/NC_000913.fna"
    home_file = "/home/pat/ecoli/NC_000913.fna"
    with open(lab_file if at_lab else home_file) as f:
        genome = "".join([line.strip() for line in f.readlines()[1:]])
    return "".join(g for g in genome if g in "ATGC") # contains other iupac symbols

def random_substring(xs,k):
    i = random.randint(0,len(xs)-k)
    return xs[i:i+k]

def subst(xs,ys,i):
    """Substitute substring ys in xs, starting at i"""
    if not (type(ys) is list or type(ys) is str):
        ys = [ys]
    return xs[:i] + ys + xs[i+len(ys):]

def cumsum_ref(xs):
    return [sum(xs[:i+1]) for i in range(len(xs))]

def cumsum(xs):
    acc = 0
    acc_list = []
    for x in xs:
        acc += x
        acc_list.append(acc)
    return acc_list

def inverse_cdf_sample_reference(xs,ps):
    """Sample from xs according to probability distribution ps"""
    PS = cumsum(ps)
    r = random.random()
    P,x = min(filter(lambda (P,x):P > r,zip(PS,xs)))
    return x

def inverse_cdf_sample_reference2(xs,ps):
    """Sample from xs according to probability distribution ps"""
    PS = fast_cumsum(ps)
    r = random.random()
    P,x = min(filter(lambda (P,x):P > r,zip(PS,xs)))
    for P,x in zip(PS,xs): #slightly slower!
        if P > r:
            return x

def inverse_cdf_reference3(xs,ps):
    """Sample from xs according to probability distribution ps"""
    PS = fast_cumsum(ps)
    r = random.random()
    P,x = min(filter(lambda (P,x):P > r,zip(PS,xs)))
    return x

def inverse_cdf_sample(xs,ps,normalized=True):
    r = random.random()
    if not normalized:
        r *= sum(ps)
    acc = 0
    for x,p in zip(xs,ps):
        acc += p
        if acc > r:
            return x
    raise Exception("failed to return on:",xs,ps)

def pl(f,xs):
    """A convenience function for plotting.
    Usage: plt.plot(*pl(f,xs))"""
    return [xs,map(f,xs)]

def argmax(xs):
    i,x = max(enumerate(xs),key= lambda (i,x):x)
    return i

def argmin(xs):
    i,x = min(enumerate(xs),key= lambda (i,x):x)
    return i

def generate_greedy_motif_with_ic(desired_ic,epsilon,num_seqs,length,verbose=False):
    motif = [random_site(length) for i in range(num_seqs)]
    ic = motif_ic(motif)
    while(abs(desired_ic - ic) > epsilon):
        motif_prime = motif[:]
        n = random.randrange(num_seqs)
        l = random.randrange(length)
        motif_prime[n] = subst(motif_prime[n],random.choice("ACGT"),l)
        ic_prime = motif_ic(motif_prime)
        if abs(ic_prime - desired_ic) < abs(ic - desired_ic):
            motif = motif_prime
            ic = motif_ic(motif)
        if verbose:
            print ic
    return motif

def sa_motif_with_desired_ic(desired_ic,epsilon,num_seqs,length,verbose=False):
    f = lambda motif:abs((motif_ic(motif))-desired_ic)
    proposal = mutate_motif
    x0=random_motif(length,num_seqs)
    return anneal(f,proposal,x0,stopping_crit=epsilon,verbose=verbose)
    
def first(x):
    return x[0]

def second(x):
    return x[1]

def third(x):
    return x[2]

def omit(xs,i):
    return [x for j,x in enumerate(xs) if not i == j]

def cv(data,k=10,randomize=True):
    """Return the dataset chunked into k tuples of the form
    (training_set, test_set)"""
    data_copy = data[:]
    if randomize:
        random.shuffle(data_copy)
    chunks = group_into(data_copy,k)
    return [(concat(omit(chunks,i)),chunks[i]) for i in range(k)]

def acf(xs,min_lag=None,max_lag=None,verbose=False):
    """Compute the auto-correlation naively"""
    n = len(xs)
    if min_lag is None:
        min_lag = 0
    if max_lag is None:
        max_lag = n/2
    mu = mean(xs)
    sigma_sq = variance(xs)
    ts = verbose_gen(xrange(min_lag,max_lag)) if verbose else xrange(min_lag,max_lag)
    return [mean([(xs[i] - mu) * (xs[i+t]-mu)/sigma_sq for i in range(n-t)])
            for t in ts]

def rslice(xs,js):
    return [xs[j] for j in js]

def exponential(z):
        a = z.real
        b = z.imag
        return exp(a)*(cos(b) + 1j * sin(b))

def dft(xs):
    N = float(len(xs))
    return [sum(x_n*(exponential(-1j*2*pi*k*n/N))
                for (n,x_n) in enumerate(xs))
            for k in range(int(N))]

def inv_dft(xs):
    N = float(len(xs))
    return [sum(x_n*exponential(1j*2*pi*k*n/N) for (n,x_n) in enumerate(xs))/N
            for k in range(int(N))]

def convolve(xs,ys):
    XS = dft(xs)
    YS = dft(ys)
    return inv_dft([X*Y for (X,Y) in zip(XS,YS)])

def circular_rolling_average(xs,k):
    """return circular rolling average of window of /radius/ k centered
    about x"""
    n = len(xs)
    ys = xs[-k:] + xs + xs[:k]
    return [mean(ys[(i-k):(i+k+1)]) for i in xrange(k,n + k)]

def weiner_deconvolution(ys):
    # Supposing y = h*x + n
    # where x is the true signal,
    # h is the impulse response
    pass

def dpoisson(k,lamb):
    """Poisson density function"""
    return lamb**k*exp(-lamb)/fac(k)

def rpoisson(lamb):
    """Poisson sampling.  (Knuth vol. 2)"""
    L = exp(-lamb)
    k = 0
    p = 1
    while p > L:
        k = k + 1
        u = random.random()
        p *= u
    return k - 1

def poisson_model(xs):
    """Return a poisson model of xs"""
    lamb = mean(xs)
    return [rpoisson(lamb) for x in xs]
    
def true_positives_ref(predicted,actual):
    return sum(zipWith(lambda p,a:p==1 and a == 1,predicted,actual))

def true_positives(predicted,actual):
    count = 0
    for p,a in zip(predicted,actual):
        count += p*a
    return count

def true_negatives(predicted,actual):
    return sum(zipWith(lambda p,a:p==0 and a == 0,predicted,actual))

def false_positives_ref(predicted,actual):
    return sum(zipWith(lambda p,a:p==1 and a == 0,predicted,actual))

def false_positives(predicted,actual):
    count = 0
    for p,a in zip(predicted,actual):
        count += p*(1-a)
    return count

def false_negatives(predicted,actual):
    return sum(zipWith(lambda p,a:p==0 and a == 1,predicted,actual))

def precision(predicted,actual):
    tp = true_positives(predicted,actual)
    fp = false_positives(predicted,actual)
    return tp/float(tp + fp + epsilon)

def recall(predicted,actual):
    tp = true_positives(predicted,actual)
    fn = false_negatives(predicted,actual)
    return tp/float(tp + fn + epsilon)

def accuracy(predicted,actual):
    tp = true_positives(predicted,actual)
    tn = true_negatives(predicted,actual)
    fp = false_positives(predicted,actual)
    fn = false_negatives(predicted,actual)
    return (tp + tn)/float(tp + tn + fp + fn)

def f_score(predicted,actual):
    p = precision(predicted,actual)
    r = recall(predicted,actual)
    return 2 * p*r/(p + r + epsilon)

def roc_curve_ref(positives,negatives,thetas=None,color=None):
    """Given a set of positive scores, a set of negative scores, and a
    range of theta values, plot a ROC curve for f on instances.

    """
    positives = np.array(positives)
    negatives = np.array(negatives)
    instances = np.concatenate((positives,negatives))
    if thetas is None:
        thetas = sorted(set(instances))
    if color is None:
        color= 'b'
    tprs = []
    fprs = []
    Np = len(positives)
    Nn = len(negatives)
    N = float(Np + Nn)
    actual = np.array(([1]*Np) + ([0]*Nn))
    for theta in thetas:
        predicted = 1*(instances > theta)
        pred_positive = float(sum(predicted))
        tpr = np.dot(predicted,actual)/float(Np)
        fpr = np.dot(predicted,1-actual)/float(Nn)
        tprs.append(tpr)
        fprs.append(fpr)
        print theta,tpr,fpr
    plt.plot(fprs,tprs,color=color)
    plt.xlabel("FPR")
    plt.ylabel("TPR")

def roc_curve(positives,negatives,color=None,annotate=False, return_auc=False):
    """Given a set of positive scores and a set of negative scores, plot a
    ROC curve.

    Implements Algorithm 2 of:

    Fawcett, T. (2004), 'ROC graphs: Notes and practical
    considerations for researchers', ReCALL 31 (HPL-2003-4) , 1--38 .

    """
    instances = sorted([(x,0) for x in negatives] + [(x,1) for x in positives],
                       key = lambda (x,cls):x,
                       reverse=True)
    i = 0
    if color is None:
        color= 'b'
    tprs = []
    fprs = []
    thetas = []
    Np = float(len(positives))
    Nn = float(len(negatives))
    tp = 0
    fp = 0
    theta = min([x for (x,cls) in instances]) - 1
    theta_prev = theta
    for theta,cls in instances:
        if theta != theta_prev:
            tprs.append(tp/Np)
            fprs.append(fp/Nn)
            thetas.append(theta)
            theta_prev = theta
        if cls == 1:
            tp += 1
        else:
            fp += 1
    tprs.append(tp/Np)
    fprs.append(fp/Nn)
    thetas.append(theta)
    plt.plot(fprs,tprs,color=color)
    if annotate:
        theta_labels = ["%e" % theta for theta in thetas]
        annotations = unique(zip(fprs,tprs,theta_labels))
        modulus = len(annotations)/10
        print "%s unique annotations" % len(annotations)
        for i,(fpr,tpr,theta) in enumerate(annotations):
            if i % modulus == 0:
                plt.annotate(theta,xy=(fpr,tpr),xytext=(-20,20), textcoords = 'offset points',
                             ha = 'right', va = 'bottom',
                             bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                             arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot([0,1],[0,1],linestyle='--')
    auc = sum(1/2.0 * (y1+y0)*(x1-x0) for ((x0,y0),(x1,y1)) in pairs(zip(fprs,tprs)))
    return fprs,tprs,thetas, auc
    
def sliding_window(seq,w,verbose=False):
    i = 0
    n = len(seq)
    while i < n - w + 1:
        if verbose:
            if i % verbose == 0:
                print i
        yield seq[i:i+w]
        i += 1

def consensus(motif):
    """Return the consensus of a motif"""
    cols = transpose(motif)
    return "".join([char for (char,count) in
                    map(lambda col:max(Counter(col).items(),key=lambda (b,c):c),
                        cols)])

def random_model(xs):
    mu = mean(xs)
    sigma = sd(xs)
    return [random.gauss(mu,sigma) for x in xs]

def qqplot(xs,ys=None):    
    if ys is None:
        ys = normal_model(xs)
    min_val = min(min(xs), min(ys))
    max_val = max(max(xs), max(ys))
    plt.scatter(sorted(xs),sorted(ys))
    print "Mann-Whitney U test:",mannwhitneyu(xs,ys)
    plt.plot([min_val,max_val],[min_val,max_val])
    
def head(xs, p=iota):
    """Take first element of xs, optionally satisfying predicate p"""
    filtered_xs = filter(p, xs)
    return filtered_xs[0] if filtered_xs else []

def find(p,xs):
    """return first element of xs satisfying p, or None if none do"""
    for x in xs:
        if p(x):
            return x
    return None

def binary_find(p,xs):
    """find first x satisfying p, assuming monotonicity"""
    lo = 0
    hi = len(xs)
    plo = p(lo)
    phi = p(hi)
    if not phi:
        return None
    while hi - lo > 1:
        guess = int((lo + hi)/2)
        pguess = p(guess)
        print lo,plo,hi,phi,guess,pguess
        if pguess:
            hi = guess
            phi = pguess
        else:
            lo = guess
            plo = pguess
    print "lo:",plo
    print "hi:",phi
    print "guess:",pguess
    return guess
    
def product_ref(xs):
    """multiply elements of list.  Python doesn't implement TCO so
    this implementation is not safe for production.  See product."""
    return foldl(lambda x,y:x*y,1,xs)

def prod(xs):
    acc = 1
    for x in xs:
        acc *= x
    return acc

def unflip_motif(motif):
    """Given a collection of possibly reverse complemented sites,unflip them"""
    from sufficache import PSSM
    mutable_motif = motif[:]
    for i,site in enumerate(motif):
        loo_motif = [s for (j,s) in enumerate(motif) if not i == j]
        pssm = PSSM(loo_motif)
        fd_score = pssm.score(site,both_strands=False)
        bk_score = pssm.score(wc(site),both_strands=False)
        print site
        print fd_score,bk_score
        if bk_score > fd_score:
            mutable_motif[i] = wc(site)
    return mutable_motif

def interpolate(start,stop,steps):
    return [start + i*(stop-start)/float(steps-1) for i in range(steps)]

def maybesave(filename):
    """
    Convenience function for plots.
    """
    if filename:
        plt.savefig(filename,dpi=400)
        plt.close()
    else:
        plt.show()

def mh(f,proposal,x0,dprop=None,iterations=50000,every=1,verbose=0,use_log=False,capture_state=lambda x:x,modulus=1000,cache=True,return_ar=False):
    """General purpose Metropolis-Hastings sampler.  If use_log is
    true, assume that f is actually log(f)"""
    if dprop is None:
        if verbose > 0:
            print "Warning: using M-H without proposal density: ensure that proposal is symmetric!"
        dprop = lambda x_new,x:1
    x = x0
    xs = [x]
    acceptances = 0
    proposed_improvements = 0
    try:
        for it in xrange(iterations):
            if it == 0 or not cache: # if not caching, reevaluate every time
                fx = f(x)
            if it % modulus == 0 and verbose:
                print it,fx
            x_new = proposal(x)
            fx_new = f(x_new)
            if not use_log:
                prop_ratio = dprop(x,x_new)/dprop(x_new,x) if dprop else 1
                ratio = fx_new/fx*prop_ratio
                r = random.random() 
            else: #using log
                prop_ratio = dprop(x,x_new) - dprop(x_new,x) # assume density proposal is log too!
                ratio = (fx_new - fx) + prop_ratio
                r = log(random.random())
            if verbose and it % verbose == 0:
                comp = cmp(fx_new,fx)
                characterization = {1:"improvement",0:"stasis",-1:"worsening"}[comp]
                if comp == 1:
                    proposed_improvements += 1
                print it,"fx:",fx,"fx_new:",fx_new,"ratio:",ratio,characterization,"r:",r,\
                       "accept" if ratio > r else "decline","acceptance ratio:",acceptances/float(max(it,1))
            if ratio > r:
                x = x_new
                fx = fx_new
                acceptances += 1
            if it % every == 0:
                xs.append(capture_state(x))
        if verbose:
            print "Proposed improvement ratio:",proposed_improvements/float(iterations)
        if verbose:
            print "Acceptance Ratio:",acceptances/float(iterations)
        if return_ar:
            return acceptances/float(iterations)
        else:
            return xs
    except KeyboardInterrupt:
        return xs

def anneal(f,proposal,x0,iterations=50000,T0=1,tf=0,k=1,verbose=False,stopping_crit=None,
           return_trajectory=False,raise_exception_on_failure=False):
    """General purpose simulated annealing: minimize f, returning
    trajectory of xs.  stopping_crit is a constant such that x is
    returned if f(x) < stopping_crit.  k is a cooling constant; higher
    values of k result in faster cooling."""
    x = x0
    x_min = x
    f_min = f(x)
    if return_trajectory:
        xs = [x]
    fx = f(x)
    acceptances = 0
    def get_temp(it):
        """Return temp for given iteration"""
        return tf + T0*exp(-k*it)
    for i in xrange(iterations):
        T = get_temp(i)
        if i % 1000 == 0:
            print i,fx,T
        x_new = proposal(x)
        fx_new = f(x_new)
        T = get_temp(i)
        if T == 0:
            break
        log_ratio = (1/T * (fx-fx_new))
        if verbose:
            print "fx:",fx,"fx_new:",fx_new,"log_ratio:",log_ratio,"Temperature:",T
        if log_ratio > log(random.random()):
            x = x_new
            fx = fx_new
            acceptances += 1
            if fx < f_min:
                x_min = x
        if return_trajectory:
            xs.append(x)
        if fx < stopping_crit:
            print "Acceptance Ratio:",acceptances/float(iterations)
            return xs if return_trajectory else x_min
    # if we did not ever satisfy the stopping criterion...
    if raise_exception_on_failure:
        raise(Exception("Failed to anneal"))
    else:
        return xs if return_trajectory else x_min
    

def gini(xs):
    ys = sorted(xs)
    n = float(len(ys))
    if sum(ys) == 0:
        return 0
    else:
        return (2*sum((i+1)*y for i,y in enumerate(ys)))/(n*sum(ys)) - (n+1)/n

def motif_gini(motif,correct=False):
    """Return the gini coefficient of the column ics"""
    return gini(columnwise_ic(motif,correct=correct))
    
def motif_kl(motif1,motif2,pseudocount=1/4.0):
    """Return Kullbeck-Leibler divergence of two motifs, assuming
    independence between columns"""
    n = float(len(motif1))
    assert(n == len(motif2) and len(motif1[0]) == len(motif2[0]))
    ps = [[(col.count(b) + pseudocount)/n for b in "ACGT"] for col in transpose(motif1)]
    qs = [[(col.count(b) + pseudocount)/n for b in "ACGT"] for col in transpose(motif2)]
    return sum([pj*log2(pj/qj)
                for p,q in zip(ps,qs)
                for pj,qj in zip(p,q)
                if pj or qj])

def column_distance(col1,col2):
    m = float(len(col1))
    n = float(len(col2))
    assert m == n
    def freqs(col):
        return (col.count('A')/n,col.count('C')/n,col.count('G')/n,col.count('T')/n)
    freqs1 = freqs(col1)
    freqs2 = freqs(col2)
    return 2*n**(1/2.0)*acos((sum((p1*p2)**(1/2.0) for p1,p2 in zip(freqs1,freqs2)))-10**-50)

def motif_distance(motif1,motif2):
    cols1 = transpose(motif1)
    cols2 = transpose(motif2)
    return norm(zipWith(column_distance,cols1,cols2))

def motif_hamming_distance(m1,m2):
    return sum(zipWith(hamming,m1,m2))
    
def pred_obs(xys,label=None,color='b',show=True):
    xs,ys = transpose(xys)
    minval = min(xs+ys)
    maxval = max(xs+ys)
    plt.scatter(xs,ys,color=color)
    plt.plot([minval,maxval],[minval,maxval])
    if show:
        plt.show()

def make_pssm(seqs,pseudocount=1):
    cols = transpose(seqs)
    N = float(len(seqs))
    return [[log2((col.count(b)+pseudocount)/(N+4.0*pseudocount)) - log2(0.25) for b in "ACGT"] for col in cols]
    
def score_seq_ns(matrix,seq,ns=False):
    """Score a sequence with a motif."""
    base_dict = {'A':0,'C':1,'G':2,'T':3}
    ns_binding_const = -8 #kbt
    #specific_binding = sum([row[base_dict[b]] for row,b in zip(matrix,seq)])
    specific_binding = 0
    for i in range(len(matrix)):        
        specific_binding += matrix[i][base_dict[seq[i]]]
    if ns:
        return log(exp(-beta*specific_binding) + exp(-beta*ns_binding_const))/-beta
    else:
        return specific_binding

def score_seq(matrix,seq):
    #base_dict = {'A':0,'C':1,'G':2,'T':3}
    def base_dict(b):
        if b <= "C":
            if b == "A":
                return 0
            else:
                return 1
        elif b == "G":
            return 2
        else:
            return 3
    ans = 0
    for i in xrange(len(seq)):
        ans += matrix[i][base_dict(seq[i])]
    return ans

def seq_scorer(matrix):
    """accept matrix, return a function scoring sites"""
    # when score_seq JUST ISN'T FAST ENOUGH
    base_dicts = [{b:row[j] for j,b in enumerate("ACGT")} for row in matrix]
    def f(site):
        ans = 0
        for i in xrange(len(site)):
            ans += base_dicts[i][site[i]]
        return ans
    return f
        

        
def score_genome(matrix,genome,ns=False):
    w = len(matrix)
    L = len(genome)
    return [score_seq(matrix,genome[i:i+w],ns=ns) for i in range(L-w+1)]
    
def uncurry(f):
    """
    (a -> b -> c) -> (a, b) -> c
    """
    return lambda (a,b):f(a,b)

def mapdict(d,xs):
    return [d[x] for x in xs]

def simplex_sample(n):
    """Sample uniformly from the probability simplex with n-components"""
    xs = [0] + sorted([random.random() for i in range(n-1)]) + [1]
    diffs = [x2-x1 for (x1,x2) in pairs(xs)]
    return diffs

def how_many(p,xs):
    return len(filter(p,xs))

def dnorm(x,mu=0,sigma=1):
    return 1/(sigma*sqrt(2*pi))*exp(-(x-mu)**2/float(2*sigma**2))

def take(n,xs):
    """Return first n items of xs, or all if len(xs) < n"""
    # We do this in a slightly stupid way in order to make this work with generators
    i = 0
    taken = []
    for x in xs:
        taken.append(x)
        i += 1
        if i == n:
            break
    return taken

def count(p,xs):
    """Count number of xs satisfying p"""
    return len(filter(p,xs))

def inverse_cdf_sampler(xs,ps):
    """make a bintree for Sampling from discrete distribution ps over set xs"""
    cum_ps = cumsum(ps)
    def sampler():
        r = random.random()
        i = bisect.bisect_left(cum_ps,r)
        return xs[i]
    return sampler

def fisher_dist(ps,qs):
    return acos(sum([sqrt(p*q) for p,q in zip(ps,qs)]))

def verb(x,verbose=False):
    if verbose:
        print x

def report_vars(var_string):
    print ", ".join(var_name + ":" + str(eval(var_name))
                   for var_name in var_string.split())

def kmers(n):
    return itertools.product(*["ACGT" for i in range(n)])

def prod_dep(xs):
    return reduce(lambda x,y:x*y,xs)

def sample_until_ref(p,sampler,n):
    xs = []
    trials = 0
    while len(xs) < n:
        x = sampler()
        if p(x):
            xs.append(x)
        trials += 1
    return xs

def sample_until(p,sampler,n,progress_bar=True):
    """return n samples from sampler satisfying predicate p"""
    def gen():
        while True:
            x = sampler()
            if p(x):
                return x
    pb = trange if progress_bar else xrange #optional progress bar
    return [gen() for i in pb(n)]

def cohens_d(xs,ys):
    return (mean(xs) - mean(ys)) / (sqrt((sd(xs) ** 2 + sd(ys) ** 2) / 2))

def compose(f,g):
    return lambda x:f(g(x))

def diffs(xs):
    return [x2-x1 for (x1,x2) in pairs(xs)]
        
def param_scan(f,xs,ys):
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = plt.subplots()
    mat = [[f(x,y) for x in xs] for y in tqdm(ys)]
    plt.imshow(mat,interpolation='none')
    plt.xticks(range(len(xs)),xs,rotation=90)
    plt.yticks(range(len(xs)),ys)
    plt.colorbar()

def kde(xs,sigma=1):
    def f(xp):
        return mean(dnorm(xp,mu=x,sigma=sigma) for x in xs)
    return f

def kde_regress(xs, ys, sigma=None, monotonic=False):
    if monotonic:
        def is_monotonic(f):
            yhats = map(f,sorted(xs))
            return -1 + 2 * all(y1 <= y2 for y1,y2 in pairs(yhats))
        g = lambda sigma: is_monotonic(kde_regress(xs,ys,sigma=sigma))
        lb = ub = 0.01
        while g(ub) < 0:
            ub *= 2
            print "ub:", ub
        sigma = bisect_interval(g, lb, ub, xtol=10**-3)
        return kde_regress(xs,ys,sigma=sigma)
    print "sigma:",sigma
    if sigma is None:
        sigma = 3*(max(xs) - min(xs))/float(len(xs))
        print "sigma:",sigma
    def f(xp):
        ds = [dnorm(xp,mu=x,sigma=sigma) for x,y in zip(xs,ys)]
        Z = sum(ds)
        return sum(y*d for d,y in zip(ds,ys))/Z
    return f

def gelman_rubin(chains):
    N = len(chains[0])
    burned_chains = [chain[N/2:] for chain in chains] # eliminate burn-in
    # now split each one in half
    halved_chains = concat([(chain[:len(chain)/2],chain[len(chain)/2:]) for chain in burned_chains])
    min_len = min(map(len,halved_chains))
    halved_chains = [hc[:min_len] for hc in halved_chains]
    m = len(halved_chains)
    n = len(halved_chains[0])
    psi = np.matrix(halved_chains).transpose()
    psi_bar = np.mean(psi)
    B = n/float(m-1)*sum((np.mean(psi[:,j]) - psi_bar)**2 for j in range(m))
    def sj_sq(j):
        psi_j = np.mean(psi[:,j])
        return 1.0/(n-1)*sum((psi[i,j] - psi_j)**2 for i in range(n))
    W = 1.0/m * sum(sj_sq(j)for j in range(m))
    var_hat_plus = (n-1)/float(n)*W + 1.0/n * B
    R_hat = sqrt(var_hat_plus/W)
    def V(t):
        return 1.0/(m*(n-t))*sum((psi[i,j] - psi[i-t,j])**2 for i in range(t+1,n) for j in range(m))
    def rho_hat(t):
        return 1 - V(t)/(2*var_hat_plus)
    crit = lambda t:(t % 2 == 1) and rho_hat(t+1) + rho_hat(t+2) < 0
    #T = find(lambda t:(t % 2 == 1) and rho_hat(t+1) + rho_hat(t+2) < 0,range(n-1))
    T = binary_find(crit,range(n))
    if not T is None:
        neff = m*n/(1 + 2*sum(rho_hat(t) for t in range(1,T+1)))
    else:
        neff = None
    return R_hat,neff

def scatter(xs, ys, line_color='black', color='b'):
    plt.scatter(xs,ys,color=color)
    minval = min(map(min,[xs,ys]))
    maxval = max(map(max,[xs,ys]))
    plt.plot([minval,maxval],[minval,maxval],linestyle='--',color=line_color)
    return pearsonr(xs,ys)
    #print spearmanr(xs,ys)

def format_params(param_names_str):
    """given string of parameters 'x y z' return string of form:
    'x = $x, y=$y, z = $z' """
    var_names = param_names_str.split()
    param_template = ", ".join("{0}=%({0})s".format(v) for v in var_names)
    param_string = param_template % globals()
    return param_string

def logmod(x, base=10):
    return sign(x)*log(abs(x) + 1, base)

def pearson_na(xs, ys):
    return pearsonr(*transpose(filter(lambda (x,y):not (pd.isnull(x) or
                                                         pd.isnull(y)),
                                       zip(xs,ys))))
def log10(x):
    return log(x, 10)

def logsum(log_xs):
    """given log_xs = map(log,xs), return log(sum(xs))"""
    a = max(log_xs)
    return a + log(sum(exp(log_x-a) for log_x in log_xs))

def robbins_munro(f, x0, iterations=1000, verbose=False, return_hist=False):
    """given f, a noisy function assumed to be monotonically increasing,
find the root (f(x) == 0), starting from guess x0"""
    x = x0
    x_hist, y_hist = [], []
    for i in range(1, iterations + 1):
        y = f(x)
        if return_hist:
            x_hist.append(x)
            y_hist.append(y)
        x += 1.0/i * (-y)
        if verbose:
            print i, x, y
    if not return_hist:
        return x
    else:
        return x, x_hist, y_hist

