import editdistance as ed
import operator
import copy

INSERT = 'insert'
DELETE = 'delete'
EQUAL = 'equal'
REPLACE = 'replace'

phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl",
                "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh",
                "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng",
                "nx", "l", "r", "w", "y", "hh", "hv", "el", "iy", "ih",
                "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow",
                "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

phonemes2index = {k: (v+2) for v, k in enumerate(phonemes)}
index2phonemes = {(v+2): k for v, k in enumerate(phonemes)}

phoneme_reduce_mapping = {"b": "b", "bcl": "h#", "d": "d", "dcl": "h#", "g": "g",
                          "gcl": "h#", "p": "p", "pcl": "h#", "t": "t", "tcl": "h#",
                          "k": "k", "kcl": "h#", "dx": "dx", "q": "q", "jh": "jh",
                          "ch": "ch", "s": "s", "sh": "sh", "z": "z", "zh": "sh",
                          "f": "f", "th": "th", "v": "v", "dh": "dh", "m": "m",
                          "n": "n", "ng": "ng", "em": "m", "en": "n", "eng": "ng",
                          "nx": "n", "l": "l", "r": "r", "w": "w", "y": "y",
                          "hh": "hh", "hv": "hh", "el": "l", "iy": "iy", "ih": "ih",
                          "eh": "eh", "ey": "ey", "ae": "ae", "aa": "aa", "aw": "aw",
                          "ay": "ay", "ah": "ah", "ao": "aa", "oy": "oy", "ow": "ow",
                          "uh": "uh", "uw": "uw", "ux": "uw", "er": "er", "ax": "ah",
                          "ix": "ih", "axr": "er", "ax-h": "ah", "pau": "h#", "epi": "h#",
                          "h#": "h#"}

reduce_phonemes = set([phoneme_reduce_mapping[ch] for ch in phonemes])
reduce_phonemes.remove('q')
reduce_phonemes2index = {ch: phonemes2index[ch] for ch in reduce_phonemes}
index2reduce_phonemes = {phonemes2index[ch]: ch for ch in reduce_phonemes}


def lowest_cost_action(ic, dc, sc, cost):
    """Given the following values, choose the action (insertion, deletion,
    or substitution), that results in the lowest cost (ties are broken using
    the 'match' score).  This is used within the dynamic programming algorithm.
    * ic - insertion cost
    * dc - deletion cost
    * sc - substitution cost
    * im - insertion match (score)
    * dm - deletion match (score)
    * sm - substitution match (score)
    """
    best_action = None
    min_cost = min(ic, dc, sc)
    if min_cost == sc and cost == 0:
        best_action = EQUAL
    elif min_cost == sc and cost == 1:
        best_action = REPLACE
    elif min_cost == ic:
        best_action = INSERT
    elif min_cost == dc:
        best_action = DELETE
    return best_action


def edit_distance(seq1, seq2, action_function=lowest_cost_action, test=operator.eq):
    """Computes the edit distance between the two given sequences.
    This uses the relatively fast method that only constructs
    two columns of the 2d array for edits.  This function actually uses four columns
    because we track the number of matches too.
    """
    m = len(seq1)
    n = len(seq2)
    # Special, easy cases:
    if seq1 == seq2:
        return 0, n
    if m == 0:
        return n, 0
    if n == 0:
        return m, 0
    v0 = [0] * (n + 1)     # The two 'error' columns
    v1 = [0] * (n + 1)
    for i in range(1, n + 1):
        v0[i] = i
    for i in range(1, m + 1):
        v1[0] = i
        for j in range(1, n + 1):
            cost = 0 if test(seq1[i - 1], seq2[j - 1]) else 1
            # The costs
            ins_cost = v1[j - 1] + 1
            del_cost = v0[j] + 1
            sub_cost = v0[j - 1] + cost

            action = action_function(ins_cost, del_cost, sub_cost, cost)

            if action in [EQUAL, REPLACE]:
                v1[j] = sub_cost
            elif action == INSERT:
                v1[j] = ins_cost
            elif action == DELETE:
                v1[j] = del_cost
            else:
                raise Exception('Invalid dynamic programming option returned!')
                # Copy the columns over
        for k in range(0, n + 1):
            v0[k] = v1[k]
    return v1[n]


def edit_distance_by_phoneme(seq1, seq2, action_function=lowest_cost_action, test=operator.eq):
    """Computes the edit distance between the two given sequences.
    This uses the relatively fast method that only constructs
    two columns of the 2d array for edits.  This function actually uses four columns
    because we track the number of matches too.
    """
    def get_class(idx):
        reduce_idx2broad_class_idx = {2: 0, 4: 0, 6: 0, 8: 0, 10: 0, 12: 0, 14: 0,  # Stops      b, d, g, p, t, k, dx, q
                                      16: 1, 17: 1,  # Affricate  jh, ch
                                      18: 2, 19: 2, 20: 2, 22: 2, 23: 2, 24: 2, 25: 2,  # Fricative  s, sh, z, zh, f, th, v, dh
                                      33: 3, 34: 3, 35: 3, 36: 3, 37: 3,  # Glides     l, r, w, y, hh, hv, el
                                      26: 4, 27: 4, 28: 4,  # Nasals     m, n, ng, em, en, eng, nx
                                      40: 5, 41: 5, 42: 5, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5, 50: 5, 51: 5, 52: 5, 53: 5, 55: 5,  # Vowels     iy, ih, eh, ey, ae, aa, aw, ay, ah, ao, oy, ow, uh, uw, ux, er, ax, ix, axr, ax-h
                                      62: 6}  # Others     pau, epi, h#
        return reduce_idx2broad_class_idx[idx]

    m = len(seq1)
    n = len(seq2)
    # Special, easy cases:
    if seq1 == seq2:
        return 0, n
    if m == 0:
        return n, 0
    if n == 0:
        return m, 0
    v0 = [0] * (n + 1)     # The two 'error' columns
    v1 = [0] * (n + 1)
    p0 = [[0]*7 for _ in range(n+1)]
    p1 = [[0]*7 for _ in range(n+1)]

    for i in range(1, n + 1):
        v0[i] = i
        p0[i][get_class(seq2[i - 1])] = i

    for i in range(1, m + 1):
        v1[0] = i
        p1[0][get_class(seq1[i - 1])] = i
        for j in range(1, n + 1):
            phonetic_class = get_class(seq2[j - 1])

            cost = 0 if test(seq1[i - 1], seq2[j - 1]) else 1
            # The costs
            ins_cost = v1[j - 1] + 1
            del_cost = v0[j] + 1
            sub_cost = v0[j - 1] + cost

            action = action_function(ins_cost, del_cost, sub_cost, cost)

            if action in [EQUAL, REPLACE]:
                v1[j] = sub_cost
                for k in range(7):
                    if k == phonetic_class:
                        p1[j][k] = p0[j - 1][k] + cost
                    else:
                        p1[j][k] = p0[j - 1][k]
            elif action == INSERT:
                v1[j] = ins_cost
                p1[j][phonetic_class] = ins_cost
            elif action == DELETE:
                v1[j] = del_cost
                p1[j][phonetic_class] = del_cost
            else:
                raise Exception('Invalid dynamic programming option returned!')
                # Copy the columns over

        v0 = copy.deepcopy(v1)
        p0 = copy.deepcopy(p1)
    return v1[n]


if __name__ == '__main__':
    pred = [2, 16, 8, 18]
    true = [2, 17, 4, 18]
    print(ed.eval(pred, true)/len(true))
    print(edit_distance(pred, true)/len(true))
    print(edit_distance_by_phoneme(pred, true)/len(true))
