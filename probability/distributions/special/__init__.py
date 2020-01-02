try:
    import numba
    from probability.distributions.special import prob_bb_greater_exact as prob_bb_greater_exact
except ModuleNotFoundError:
    from probability.distributions.special import prob_bb_greater_exact as prob_bb_greater_exact
