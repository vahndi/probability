from pandas import DataFrame

from probability.discrete.joint import Joint


def fruit_boxes():
    """
    Example from Section 1.2 of "Pattern Recognition and Machine Learning"
    """
    fruit_box_data = DataFrame({
        'box': ['red'] * 8 + ['blue'] * 4,
        'fruit': ['apple'] * 2 + ['orange'] * 7 + ['apple'] * 3
    })
    # P(box,fruit)
    p_bf = Joint.from_observations(fruit_box_data)
    print(p_bf, '\n')
    # P(box)
    p_b = Joint.from_dict(
        data={'blue': 0.6, 'red': 0.4}, variables='box'
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
    p_b__f_orange = p_fb.conditional(fruit='orange')
    print(p_b__f_orange, '\n')
    # P(box=blue|fruit=orange)
    p_b_blue__f_orange = p_b__f.p(box='blue', fruit='orange')
    print(p_b_blue__f_orange, '\n')