from argparse import ArgumentParser
from classification.experiments import setup, experiments
from classification.preparation import get_classified_chunks
import itertools
from collections import Counter
import pandas as pd


def get_configs(algorithm):
    if algorithm=="all":
        feature_configs=get_all_algorithm_configs(setup.feature_transformers,
                                                  setup.feature_classifiers)
        metric_configs=get_all_algorithm_configs(setup.metric_transformers,
                                                 setup.metric_classifiers)
        setups=feature_configs+metric_configs
        return setups
    else:
        if algorithm=="metric":
            transformers=setup.metric_transformers
            classifiers=setup.metric_classifiers
        elif algorithm=="features":
            transformers=setup.feature_transformers
            classifiers=setup.feature_classifiers
        transformer=transformers[args.transformer]
        classifier=classifiers[args.classifier]
        config=setup.ExperimentConfig(transformer, classifier)
        return [config, ]


def get_all_algorithm_configs(transformers, classifiers):
    transformers=transformers.values()
    classifiers=classifiers.values()
    config_pairs=itertools.product(transformers, 
                                   classifiers)
    configs=[setup.ExperimentConfig(*params) for params in config_pairs]
    return configs


def estimator_name(config):
    return config.estimator.__class__.__name__


def display_accuracy(confmat):
    print("Confusion for %s:" % confmat.classes)
    print(confmat.confmat)
    print("Accuracy: %f" % confmat.accuracy)


def display_chunks_stats(classified_chunks):
    for cls, chunks in classified_chunks.items():
        print("%s: total %d items" % (cls, len(chunks)))


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--algorithm", required=True,
                        choices=["metric", "features", "all"], )
    parser.add_argument("--transformer")
    parser.add_argument("--classifier")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--classes", required=True, nargs="+")
    args=parser.parse_args()

    chunk_duration=pd.to_timedelta("%ds" % setup.chunk_duration_seconds)
    classified_chunks=get_classified_chunks(args.log_dir, args.classes, 
                                            chunk_duration)
    display_chunks_stats(classified_chunks)
    
    configs=get_configs(args.algorithm)
    for config in configs:
        print()
        print("Experiment: %s -> %s" %
              (estimator_name(config.transformer_config),
               estimator_name(config.classifier_config)))
        experiment=experiments.Experiment(config)
        result=experiment.run(classified_chunks)
        display_accuracy(result.confmat)
        print("Best params:", result.best_params)



