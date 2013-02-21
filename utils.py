import random
from math import sqrt,log

def log2(x):
    return log(x,2)

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
    mu = mean(xs)
    return correction * mean([(x-mu)**2 for x in xs])

def sd(xs,correct=True):
    return sqrt(variance(xs,correct=correct))

def se(xs,correct=True):
    return sd(xs,correct)/sqrt(len(xs))

def coev(xs,correct=True):
    return sd(xs,correct)/mean(xs)

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

def unique(xs):
    return list(set(xs))

def verbose_gen(xs,modulus=1):
    for i,x in enumerate(xs):
        if not i % modulus:
            print i
        yield x
        
def h(ps):
    """compute entropy (in bits) of a probability distribution ps"""
    return -sum([p * safe_log2(p) for p in ps])

def entropy(xs,correct=True,alphabet_size=None):
    """compute entropy (in bits) of a sample from a categorical
    probability distribution"""
    if alphabet_size == None:
        alphabet_size = len(set(xs)) # NB: assuming every element appears!
    ps = frequencies(xs)
    correction = ((alphabet_size - 1)/(2*log(2)*len(xs)) if correct
                  else 0) #Basharin 1959
    return h(ps) + correction

def motif_entropy(motif):
    """Return the entropy of a motif, assuming independence"""
    return sum(map(entropy,transpose(motif)))

def motif_ic(motif):
    """Return the entropy of a motif, assuming independence and a
    uniform genomic background"""
    site_length = len(motif[0])
    return 2 * site_length - motif_entropy(motif)

def mi(xs,ys):
    """Compute mutual information (in bits) of samples from two
    categorical probability distributions"""
    hx  = entropy(xs)
    hy  = entropy(ys)
    hxy = entropy(zip(xs,ys))
    return hx + hy - hxy

def permute(xs):
    """Return a random permutation of xs"""
    return sample(len(xs),xs,replace=False)

def test_permute(xs):
    permutation = permute(xs)
    return all(permutation.count(x) == xs.count(x) for x in set(xs))

def mi_permute(xs,ys,n=100,conf_int=False,p_value=False):
    """For samples xs and ys, compute an expected MI value (or
    confidence interval, or p_value for obtaining MI at least as high)
    by computing the MI of randomly permuted columns xs and ys n times"""
    replicates = [mi(permute(xs),permute(ys)) for i in range(n)]
    if conf_int:
        replicates = sorted(replicates)
        lower,upper = (replicates[int(n*.05)],replicates[int(n*.95)])
        assert lower <= upper
        return (lower,upper)
    elif p_value:
        replicates = sorted(replicates)
        mi_obs = mi(xs,ys)
        return len(filter(lambda s: s >= mi_obs,replicates))/float(n)
    else:
        return mean(replicates)

def safe_log2(x):
    """Implements log2, but defines log2(0) = 0"""
    return log(x,2) if x > 0 else 0

def group_by(xs,n):
    return [xs[i:i+n] for i in range(0,len(xs),n)]
    
def split_on(xs, pred):
    """Split xs into a list of lists each beginning with the next x
    satisfying pred, except possibly the first"""
    indices = [i for (i,v) in enumerate(xs) if pred(v)]
    return [xs[i:j] for (i,j) in zip([0]+indices,indices+[len(xs)]) if i != j]

def partition_according_to(f,xs):
    """Partition xs according to f"""
    part = []
    xsys = [(x,f(x)) for x in xs]
    yvals = set([y for (x,y) in xsys])
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

def factorial(n):
    return reduce(lambda x,y:x*y,range(1,n+1))

def choose(n,k):
    return factorial(n)/(factorial(k) * factorial(n-k))

def concat(xxs):
    return sum(xxs,[])

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
        ys = list(xs[:])
        samp = []
        for i in range(n):
            y = random.choice(ys)
            samp.append(y)
            ys.remove(y)
        return samp

def bs(xs):
    return sample(len(xs),xs,replace=True)

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
            for i in verbose_gen(range(len(A)))]

def boolean_matrix_mult(A,B):
    """Given two row-major boolean matrices as nested lists, return
    the matrix product"""
    print "in boolean matrix mult"
    return [[any(A[i][k] * B[k][j] for k in xrange(len(A[0])))
             for j in xrange(len(B[0]))]
            for i in verbose_gen(xrange(len(A)))]

def iterate(f,x,n):
    if n == 0:
        return x
    else:
        return iterate(f,f(x),n-1)

def iterate_list(f,x,n):
    if n == 0:
        return [x]
    else:
        return  [x] + iterate_list(f,f(x),n-1)
    
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

def l2(xs,ys):
    return sum(zipWith(lambda x,y:(x-y)**2,xs,ys))

def linf(xs,ys):
    return max(zipWith(lambda x,y:abs(x-y),xs,ys))

def takeWhile(p,xs):
    if not xs or not p(xs[0]):
        return []
    else:
        return [xs[0]] + takeWhile(p,xs[1:])

def distance(xs,ys):
    return sqrt(l2(xs,ys))

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
def bisect_interval(f,xmin,xmax,ymin=None,ymax=None,tolerance=1e-10,verbose=False):
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

def secant_interval(f,xmin,xmax,ymin=None,ymax=None,tolerance=1e-10):
    #print xmin,xmax,ymin,ymax
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
            return secant_interval(f,x,xmax,ymin=y,ymax=ymax,tolerance=tolerance)
        else:
            return secant_interval(f,xmin,x,ymin=ymin,ymax=y,tolerance=tolerance)

        
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
    return len(filter(lambda y:y > x,xs))/float(len(xs))

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
    """Given a list of p-values and a desired significance alpha, find
    the adjusted q-value such that a p-value less than q is expected
    to be significant at alpha, via the Benjamini-Hochberg method."""
    ps = sorted(ps)
    m = len(ps)
    ks = [k for k in range(m) if ps[k]<= k/float(m)*alpha]
    K = max(ks) if ks else None
    return ps[K] if K else None #if none are significant

def bin(scores):
    min_score = 8
    max_score = 23
    cutoffs = range(min_score,max_score,4)
    partials = [len(filter(lambda score:score >= cutoff,scores))
                for cutoff in cutoffs]
    return map(lambda(x,y):x-y,pairs(partials))

def hamming(xs,ys):
    return sum(zipWith(lambda x,y:x!=y,xs,ys))

def enumerate_mutant_neighbors(site):
    sites = [site]
    site = list(site)
    for pos in range(len(site)):
        old_base = site[pos]
        for base in "atcg":
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

def test_sorted_indices(xs):
    si = sorted_indices(xs)
    return sorted(xs) == [xs[i] for i in si]

def total_motif_mi(motif):
    cols = transpose(motif)
    return sum([mi(col1,col2) for (col1,col2) in choose2(cols)])

def random_site(n):
    return "".join(random.choice("ACGT") for i in range(n))

print "loaded utils"

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
    if not type(ys) is list:
        ys = [ys]
    return xs[:i] + ys + xs[i+len(ys):]

def cumsum(xs):
    return [sum(xs[:i+1]) for i in range(len(xs))]

def inverse_cdf_sample(xs,ps):
    """Sample from xs according to probability distribution ps"""
    PS = cumsum(ps)
    r = random.random()
    P,x = min(filter(lambda (P,x):P > r,zip(PS,xs)))
    return x

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
