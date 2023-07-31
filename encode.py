import numpy as np

def dBsequency(m, n):
    '''
    m個字母
    長度為m^n
    return de Bruijn sequence
    '''
    i, k = 1, 1
    i1 = np.ones((1, m**(n-1))) * m
    S = []
    T = np.zeros((m**(n-1), m))
    for z in range(m**n):
        j = i1[0][int(i-1)] + ((i-1) * m + 1) % (m**(n-1)) - 1
        k2 = j % m
        if k2 == 0:
            k2 = m
        T[i-1][int(k2-1)] = k
        i1[0][i-1] = i1[0][i-1] - 1
        S.append(int(k2))
        k += 1
        i = int(j)

    return S

def WordPattern(S1, S2, m1):
    '''
    S1, S2 : dBs
    m1 : numbers of word
    return structured light pattern with len(s2) * len(s1)
    '''
    m, n = len(S1), len(S2)
    W = np.zeros((n, m))
    W = np.zeros((n, m))
    W[0, :len(S1)] = S1
    for i in range(1, n):
        for j in range(m):
            W[i][j] = (W[i-1][j] + S2[i-1]) % m1
            if W[i][j] == 0:
                W[i][j] = m1
    
    return W


m = 8
s1 = dBsequency(m, 3)
s2 = dBsequency(m, 2)
w = WordPattern(s1, s2, m)
print(w.shape)
w = w - 1
for i in w:
    print(' '.join([str(int(c)) for c in i]))
np.savetxt(f'pic/pattern_{w.shape[0]}_{w.shape[1]}.txt', w)


#for decode
def X(s, m):
    '''
    x: 2 numbers in the DBsequence 
        x[0] >= x[1]
    
    Return 
        positon in the DBsequence
    '''
    if s[0] == s[1]:
        delta = 1
    else:
        delta = 0
    x1 = m**2 - s[0]**2 + (2*(s[0]-s[1])-1)*(1-delta) + 1
    return int(x1)

def Y(s, m):
    '''
    s: 3 numbers in the DBsequence 
        s[0] >= s[1] and s[0] > s[2]
        or s[0] = s[1] = s[2]
    
    Return 
        positon in the DBsequence
    '''
    if s[0] == s[1]:
        delta12 = 1
    else:
        delta12 = 0
    
    if s[1] == s[2]:
        delta23 = 1
    else:
        delta23 = 0

    pos = (m*(m+1)*(2*m+1)-s[0]*(s[0]+1)*(2*s[0]+1))/2 - (m-s[0])*(3*m+3*s[0]+1)/2 + 3*(s[0]-s[1])*(s[0]-1) + (3*(s[0]-s[2]-1)+1)*(1-delta12*delta23)+1
    return int(pos)

def D2(s, m):
    if s[0] >= s[1]:
        return X(s, m)
    elif s[0] > 1:
        return X((s[1], s[0]), m) + 1
    else:
        return X((1+s[1], s[0]), m) + 1
    
def D3(s, m):
    if s[0] == 1 and s[1] == s[2] == m:
        return m**3
    elif s[0] == s[1] == 1 and s[2] == m:
        return m**3 - 1
    elif (s[0] == s[1] == s[2]) or (s[0] >= s[1] and s[0] > s[2]):
        return Y(s, m)
    
    if s[1] >= s[2] and s[1] > s[0]:
        if s[0] == 1:
            if s[1] == s[2]:
                return Y((s[1]+1, 1, 1), m) + 2
            else:
                return Y((s[1], s[2]+1, 1), m) + 2
        else:
            return Y((s[1], s[2], s[0]), m) + 2
    
    if s[2] >= s[0] and s[2] > s[1]:
        if s[0] == s[1] == 1:
            return Y((s[2]+1, 1, 1), m) + 1
        else:
            return Y((s[2], s[0], s[1]), m) + 1
        
def decode_mod(m, reminder, a):
    # ans + a = reminder (mod m)
    ans = reminder - a
    if ans <= 0:
        ans += m
    return ans

def decode(pattern, word, m):
    #step 1 solve s1, s2
    s1, s2 = decode_mod(m, word[2], word[0]), decode_mod(m, word[4], word[2])
    print(s1, s2)
    #step 2 get x
    x = D2((s1, s2), m)
    
    #step 3 solve t1, t2, t3
    t1, t2, t3 = decode_mod(m, word[1], pattern[x][0]), decode_mod(m, word[2], pattern[x][0]), decode_mod(m, word[3], pattern[x][0])
    print(t1, t2, t3)
    #step 4 get y
    y = D3((t1, t2, t3), m)
    
    return (x, y)