import os


def get_test_data_root() -> str:
    test_root_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'data',
        'tests'
    )
    return os.path.abspath(test_root_path)
