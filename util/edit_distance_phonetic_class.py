import editdistance as ed
import operator

INSERT = 'insert'
DELETE = 'delete'
EQUAL = 'equal'
REPLACE = 'replace'


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


if __name__=='__main__':
    pred = [1, 2, 3, 4, 4, 5, 5]
    true = [1, 2, 3, 4, 5]
    print(ed.eval(pred, true)/len(true))
    print(edit_distance(pred, true)/ len(true))