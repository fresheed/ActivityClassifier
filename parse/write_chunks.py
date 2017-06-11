from classification.preparation import get_classified_chunks
from classification.experiments import setup
from argparse import ArgumentParser
import pandas as pd
import shutil
import os


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--classes", required=True, nargs="+")
    args=parser.parse_args()

    chunk_duration=pd.to_timedelta("%ds" % setup.chunk_duration_seconds)
    classified_chunks=get_classified_chunks(args.log_dir, args.classes, 
                                            chunk_duration)

    target_dir=args.target_dir
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)

    for cls, chunks in classified_chunks.items():
        for num, chunk in enumerate(chunks):
            save_path=os.path.join(target_dir, "%d_%s.chunk" % (num, cls))
            with open(save_path, "w") as out_file:
                chunk.to_csv(out_file, sep=" ")

