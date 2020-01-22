from pandas import DataFrame

from probability.pandas.joint_distribution import JointDistribution


def underline(msg):
    print()
    print(msg)
    print('=' * len(msg))


data = DataFrame({
    'box': ['red'] * 8 + ['blue'] * 4,
    'fruit': ['apple'] * 2 + ['orange'] * 7 + ['apple'] * 3
})
underline('data')
print(data)


jd = JointDistribution(data)

underline("P(fruit|box=red)")
print(jd.p('fruit', _box='red'))

underline("P(fruit,box|box=red)")
print(jd.p('fruit', 'box', _box='red'))

underline('P(fruit|box=blue)')
print(jd.p('fruit', _box='blue'))

underline('P(fruit,box|box=blue)')
print(jd.p('fruit', 'box', _box='blue'))

underline('P(fruit=orange|box=blue)')
print(jd.p(fruit='orange', _box='blue'))

underline('P(box,fruit=orange|box=blue)')
print(jd.p('box', fruit='orange', _box='blue'))

underline('P(fruit)')
print(jd.p('fruit'))

underline('P(box)')
print(jd.p('box'))
