from pathlib import Path

_tests_dir = Path(__file__)
while _tests_dir.name != 'tests':
    _tests_dir = _tests_dir.parent

DIR_TEST_DATA = _tests_dir / 'test_data'
FN_PANDAS_TESTS = DIR_TEST_DATA / 'pandas-test-data.xlsx'
