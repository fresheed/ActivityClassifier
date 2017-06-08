#! /usr/bin/python3
from argparse import ArgumentParser
import os
import shutil


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--categories", required=True, nargs="+", type=str)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    args=parser.parse_args()
    source_dir=args.source_dir
    target_dir=args.target_dir
    categories=args.categories

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)

    log_pattern=r"^%s_.*_log$"
    for for category in categories:
        pattern=log_pattern % category
        print("pattern: %s" % pattern)
        
    
