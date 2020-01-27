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
    p_b__f_apple = p_fb.condition(fruit='orange')
    print_distribution(p_b__f_apple.name, p_b__f_apple.data)


if __name__ == '__main__':

    fruit_boxes()
