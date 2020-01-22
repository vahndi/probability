from pandas import DataFrame

from probability.joint_distribution import JointDistribution

data = DataFrame({
    'box': ['red'] * 8 + ['blue'] * 4,
    'fruit': ['apple'] * 2 + ['orange'] * 7 + ['apple'] * 3
})

jd = JointDistribution(data)

print(jd.p('fruit', box='red'))
print(jd.p('fruit', 'box', box='red'))
print()

print(jd.p('fruit', box='blue'))
print(jd.p('fruit', 'box', box='blue'))
print()
print()

print(jd.p('fruit'))
print()

print(jd.p('box'))
print()
print()

print(jd.p('fruit', 'box', box='red'))
