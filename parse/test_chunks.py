from classification.features import raw
from classification.experiments.experiments import Experiment
from classification.experiments import setup
from sklearn.neural_network import MLPClassifier
from argparse import ArgumentParser
from sklearn.pipeline import Pipeline
import pandas as pd
import os


def read_chunk(path):
    frame=pd.read_csv(path, 
                      delim_whitespace=True,)
    cls=os.path.basename(path).split("_")[1]
    return frame, cls


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--chunk_dir", required=True)
    parser.add_argument("--classes", required=True, nargs="+")
    args=parser.parse_args()

    chunk_dir=args.chunk_dir
    all_files=os.listdir(chunk_dir)
    acceptable=lambda path: any([cls in path for cls in args.classes])
    used_files=filter(acceptable, all_files)
    pathes=[os.path.join(chunk_dir, path) for path in used_files]
    chunks, classes=zip(*list(map(read_chunk, pathes)))

    config=setup.ExperimentConfig(setup.feature_transformers["raw"],
                                  setup.feature_classifiers["mlp"])
    experiment=Experiment(config)
    result=experiment.run(zip(chunks, classes))
    confmat=result.confmat
    print("Confusion for %s:" % confmat.classes)
    print(confmat.confmat)
    print("Accuracy: %f" % confmat.accuracy)
    print("F1 score: %f" % confmat.f1_score)

