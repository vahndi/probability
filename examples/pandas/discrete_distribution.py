from pandas import DataFrame

from examples.pandas.shared import print_distribution
from probability.pandas.discrete_distribution import DiscreteDistribution


def fruit_boxes():
    """
    Example from Section 1.2 of "Pattern Recognition and Machine Learning"
    """
    fruit_box_data = DataFrame({
        'box': ['red'] * 8 + ['blue'] * 4,
        'fruit': ['apple'] * 2 + ['orange'] * 7 + ['apple'] * 3
    })
    # P(box,fruit)
    p_bf = DiscreteDistribution.from_dataset(fruit_box_data)
    print_distribution(p_bf.name, p_bf.data)
    # P(box)
    p_b = DiscreteDistribution.from_dict(
        data={'blue': 0.6, 'red': 0.4}, names='box'
    )
    print_distribution(p_b.name, p_b.data)
    # P(fruit|box)
    p_f__b = p_bf.condition('box')
    print_distribution(p_f__b.name, p_f__b.data)
    # P(fruit,box) = P(fruit|box) * P(box)
    p_fb = p_f__b * p_b
    print_distribution(p_fb.name, p_fb.data)
    # P(fruit)
    p_f = p_fb.margin('fruit')
    print_distribution(p_f.name, p_f.data)
    # P(box|fruit)
    p_b__f = p_fb.condition('fruit')
    print_distribution(p_b__f.name, p_b__f.data)
    # P(box|fruit=orange)
    p_b__f_orange = p_fb.condition(fruit='orange')
    print_distribution(p_b__f_orange.name, p_b__f_orange.data)


def darts():
    """
    Example from section 1.1.1 of "Bayesian Reasoning and Machine Learning"
    """
    p_region = DiscreteDistribution.from_dict({r: 1 / 20 for r in range(1, 21)}, names='region')
    print_distribution(p_region.name, p_region.data)
    p_not_20 = p_region.condition(region__ne=20)
    print_distribution(p_not_20.name, p_not_20.data)
    p_5__not_20 = p_not_20[5]
    print(p_5__not_20)


def languages_countries():
    """
    Example from section 1.1.2 of "Bayesian Reasoning and Machine Learning"
    """
    p_c = DiscreteDistribution.from_counts({
        'england': 60776238,
        'scotland': 5116900,
        'wales': 2980700
    }, 'country')
    print_distribution(p_c.name, p_c.data)
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
    print_distribution(p_l__c.name, p_l__c.data)
    p_lc = p_c * p_l__c
    print_distribution(p_lc.name, p_lc.data)


if __name__ == '__main__':

    fruit_boxes()
    darts()
    languages_countries()

