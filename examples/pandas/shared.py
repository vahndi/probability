from pandas import Series


def print_distribution(name: str, var: Series):
    print()
    print(name)
    print('=' * len(name))
    print(var)
    print('sum:', var.sum())
