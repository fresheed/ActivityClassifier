from argparse import ArgumentParser
from classification.experiments import setup, experiments
from classification.preparation import get_classified_chunks
import itertools
import pandas as pd


def get_configs(algorithm):
    if algorithm in ["all", "ci"]:
        context=setup.RunContext[algorithm.upper()]
        feature_configs=get_all_algorithm_configs(setup.feature_transformers,
                                                  setup.feature_classifiers,
                                                  context)
        metric_configs=get_all_algorithm_configs(setup.metric_transformers,
                                                 setup.metric_classifiers,
                                                 context)
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


def get_all_algorithm_configs(transformers, classifiers, context):
    context_matches=lambda conf: conf.run_context>=context
    transformers=filter(context_matches, transformers.values())
    classifiers=filter(context_matches, classifiers.values())
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
    print("F1 score: %f" % confmat.f1_score)


def display_chunks_stats(classified_chunks):
    for cls, chunks in classified_chunks.items():
        print("%s: total %d items" % (cls, len(chunks)))


def run_with_config(config):
    print()
    print("Experiment: %s -> %s" %
          (estimator_name(config.transformer_config),
           estimator_name(config.classifier_config)))
    experiment=experiments.Experiment(config)
    result=experiment.run(classified_chunks)
    return result


def display_results(results):
    ordered=reversed(sorted(results, key=lambda res: res[1].confmat.f1_score))
    for index, result in enumerate(ordered):
        config, output=result
        print()
        print("%d: Experiment: %s -> %s" %
              (index,
               estimator_name(config.transformer_config),
               estimator_name(config.classifier_config)))
        display_accuracy(output.confmat)
        print("Best params:", output.best_params)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--algorithm", required=True,
                        choices=["metric", "features", "all", "ci"], )
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
    print("Configs to run:")
    print("\n".join(map(str, configs)))

    outputs=[run_with_config(config) for config in configs]
    results=zip(configs, outputs)
    display_results(results)
    
    



