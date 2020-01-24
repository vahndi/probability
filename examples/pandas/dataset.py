from pandas import DataFrame, Series

from examples.pandas.create_data import get_fruit_box_data
from probability.pandas.dataset import DataSet


def underline(msg):
    print()
    print(msg)
    print('=' * len(msg))


data = get_fruit_box_data()


underline('data')
print(data)


jd = DataSet(data)

underline("P(fruit,box)")
print(jd.p('fruit', 'box'))

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

underline('P(box,fruit=orange)')
print(jd.p('box', fruit='orange'))

underline('P(fruit)')
print(jd.p('fruit'))

underline('P(box)')
print(jd.p('box'))


# pBr = Series(data={'red': 0.4}, name='box')
# pFo_Br = jd.p(fruit='orange', _box='red')
# pFo = jd.p(fruit='orange')
# pBr_Fo = pFo_Br * pBr / pFo
