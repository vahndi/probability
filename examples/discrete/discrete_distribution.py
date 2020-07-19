from pandas import DataFrame

from probability.discrete.discrete_distribution import DiscreteDistribution


def fruit_boxes():
    """
    Example from Section 1.2 of "Pattern Recognition and Machine Learning"
    """
    fruit_box_data = DataFrame({
        'box': ['red'] * 8 + ['blue'] * 4,
        'fruit': ['apple'] * 2 + ['orange'] * 7 + ['apple'] * 3
    })
    # P(box,fruit)
    p_bf = DiscreteDistribution.from_observations(fruit_box_data)
    print(p_bf, '\n')
    # P(box)
    p_b = DiscreteDistribution.from_dict(
        data={'blue': 0.6, 'red': 0.4}, var_names='box'
    )
    print(p_b, '\n')
    # P(fruit|box)
    p_f__b = p_bf.condition('box')
    print(p_f__b, '\n')
    # P(fruit,box) = P(fruit|box) * P(box)
    p_fb = p_f__b * p_b
    print(p_fb, '\n')
    # P(fruit)
    p_f = p_fb.margin('fruit')
    print(p_f, '\n')
    # P(box|fruit)
    p_b__f = p_fb.condition('fruit')
    print(p_b__f, '\n')
    # P(box|fruit=orange)
    p_b__f_orange = p_fb.given(fruit='orange')
    print(p_b__f_orange, '\n')
    # P(box=blue|fruit=orange)
    p_b_blue__f_orange = p_b__f.p(box='blue', fruit='orange')
    print(p_b_blue__f_orange, '\n')


def darts():
    """
    Example from section 1.1.1 of "Bayesian Reasoning and Machine Learning"
    """
    p_region = DiscreteDistribution.from_dict(
        data={r: 1 / 20 for r in range(1, 21)}, var_names='region'
    )
    print(p_region)
    p_not_20 = p_region.given(region__ne=20)
    print(p_not_20)
    p_5__not_20 = p_not_20[5]
    print(p_5__not_20)


def languages_countries():
    """
    Example from section 1.1.2 of "Bayesian Reasoning and Machine Learning"
    """
    p_c = DiscreteDistribution.from_counts({
        'england': 60_776_238,
        'scotland': 5_116_900,
        'wales': 2_980_700
    }, 'country')
    print(p_c)
    p_l__c = DiscreteDistribution.from_dict({
        ('english', 'england'): 0.95,
        ('english', 'scotland'): 0.7,
        ('english', 'wales'): 0.6,
        ('scottish', 'england'): 0.04,
        ('scottish', 'scotland'): 0.3,
        ('scottish', 'wales'): 0.0,
        ('welsh', 'england'): 0.01,
        ('welsh', 'scotland'): 0.0,
        ('welsh', 'wales'): 0.4,
    }, ['language', 'country'], 'country')
    print(p_l__c)
    p_lc = p_c * p_l__c
    print(p_lc)


if __name__ == '__main__':

    fruit_boxes()
    # darts()
    # languages_countries()

