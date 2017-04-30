#! /usr/bin/python3
from classification import try_dtw, try_corr

ALL_CLASSES=["static", "walking"]
logs_dir="parse/parsed_logs"


def main():
    for classificator in [try_dtw, try_corr]:
        print("\nUsing %s" % classificator.__name__)
        for cls in ALL_CLASSES:
            classificator.process_class(cls, logs_dir)


main()
