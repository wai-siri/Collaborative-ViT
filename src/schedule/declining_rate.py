import math

def check(a, N):
    total = 0
    for l in range(1, N + 1):
        total += math.floor(2 ** (a * (N - (l - 1))))
    return total

def declining_rate(x0, N):
    lo, hi = 0.0, 1.0
    while check(hi, N) <= x0 - 1: hi *= 2

    for _ in range(100):
        mid = (lo + hi) / 2
        if check(mid, N) <= x0 - 1:
            lo = mid
        else:
            hi = mid
    return lo