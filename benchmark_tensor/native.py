import cProfile

import numpy as np

np.random.seed(333)

LB = np.random.random(1000) - 2
RB = np.random.random(1000) + 2

NUM = 1000


def tune_range(left_border: np.ndarray, right_border: np.ndarray, num_bits: int, unify_zp: bool = False):
    level_high = 2**num_bits - 1

    if unify_zp:
        scale = (right_border - left_border) / level_high
        zero_point = -left_border / scale
        avg_zpts = np.round(np.mean(zero_point))
        qval = np.ones_like(left_border) * avg_zpts
    else:
        s = level_high / (right_border - left_border)
        fval = -left_border * s
        qval = np.round(fval)

    with np.errstate(invalid="ignore", divide="ignore"):
        ra = np.where(qval < level_high, qval / (qval - level_high) * right_border, left_border)
        rb = np.where(qval > 0.0, (qval - level_high) / qval * left_border, right_border)

    range_a = right_border - ra
    range_b = rb - left_border

    mask = np.where(range_a > range_b, 1.0, 0.0)
    inv_mask = np.abs(1.0 - mask)

    ra = mask * ra + inv_mask * left_border
    rb = inv_mask * rb + mask * right_border

    return ra, rb


with cProfile.Profile() as profiler:
    for _ in range(NUM):
        left_border = LB
        right_border = RB
        tune_range(left_border, right_border, 8, False)

profiler.print_stats()
profiler.dump_stats("native.prof")
