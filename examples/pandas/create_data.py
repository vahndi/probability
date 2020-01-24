from pandas import DataFrame


def get_fruit_box_data() -> DataFrame:
    """
    Return the data of the fruit box problem.
    """
    data = DataFrame({
        'box': ['red'] * 8 + ['blue'] * 4,
        'fruit': ['apple'] * 2 + ['orange'] * 7 + ['apple'] * 3
    })
    return data


