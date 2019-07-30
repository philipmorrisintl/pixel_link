import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Dataset converter to *.tfrecords format'
    )
    parser.add_argument(
        '--training_samples',
        dest='training_list_path',
        help='Path to *.csv file with training samples',
        type=str
    )
    parser.add_argument(
        '--test_samples',
        dest='test_list_path',
        help='Path to *.csv file with test samples',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='Folder to save converted TFRecords',
        type=str
    )
    args = parser.parse_args()
    return args
