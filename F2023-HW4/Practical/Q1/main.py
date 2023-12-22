from typing import List, Set

n = int(input())
m = int(input())

g = [[] for _ in range(n)]
rev_g = [[] for _ in range(n)]
for i in range(m):
    line = input()
    v, u = [int(i) for i in line.split("->")]
    v -= 1
    u -= 1
    g[v].append(u)
    rev_g[u].append(v)

family: List[Set[int]] = [set() for _ in range(n)]


def preprocess_families():
    seen = set()

    def get_family(v: int):
        if v in seen:
            return family[v]
        seen.add(v)
        family[v].add(v)
        for u in g[v]:
            family[v] = family[v].union(get_family(u))
        return family[v]

    for i in range(n):
        if len(rev_g[i]) == 0:
            get_family(i)


preprocess_families()


def has_active_path(st, en, givens: Set[int]):
    def check_active(a, b, c, rev_a_b: bool, rev_b_c: bool) -> bool:
        # a -> b <- c
        if rev_b_c and not rev_a_b:
            if len(family[b].intersection(givens)) > 0:
                return True
            return False

        # a -> b -> c or a <- b <- c
        if not rev_a_b ^ rev_b_c:
            return b not in givens

        # a <- b -> c
        if rev_a_b and not rev_b_c:
            return b not in givens

    tmp_seen = set()

    def traverse_all_path(v, en, before_v, rev_before_v_v: bool) -> bool:
        if v == en:
            return True
        if v in tmp_seen:
            return False
        tmp_seen.add(v)
        for u in g[v]:
            if before_v is None or check_active(before_v, v, u, rev_before_v_v, False):
                if traverse_all_path(u, en, v, False):
                    return True
        for u in rev_g[v]:
            if before_v is None or check_active(before_v, v, u, rev_before_v_v, True):
                if traverse_all_path(u, en, v, True):
                    return True
        tmp_seen.remove(v)

    return traverse_all_path(st, en, None, False)


t = int(input())
for _ in range(t):
    u = int(input())
    v = int(input())
    u -= 1
    v -= 1
    line = input()
    if line.strip() != "":
        givens = [int(i) - 1 for i in line.split(" ")]
    else:
        givens = []
    givens = set(givens)
    print(not has_active_path(u, v, givens))
