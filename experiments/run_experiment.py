from argparse import ArgumentParser
from experiments import setup, experiments
from classification.preparation import get_classified_chunks
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split


def get_configs(algorithm, transformer_code, classifier_code):
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
        transformer=transformers[transformer_code]
        classifier=classifiers[classifier_code]
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


def split_items_set(classified_chunks):
    items, classes=zip(*classified_chunks)
    train_items, test_items, train_classes, test_classes=train_test_split(items, classes, test_size=0.3, stratify=classes)
    train_set=zip(train_items, train_classes)
    test_set=zip(test_items, test_classes)
    return train_set, test_set


def estimator_name(config):
    return config.estimator.__class__.__name__


def display_accuracy(confmat):
    print("Confusion for %s:" % confmat.classes)
    print(confmat.confmat)
    print("Accuracy: %f" % confmat.accuracy)
    print("F1 score: %f" % confmat.f1_score)


def display_chunks_stats(classified_chunks):
    classes=set(pair[1] for pair in classified_chunks)
    # for cls, chunks in classified_chunks.items():
    #     print("%s: total %d items" % (cls, len(chunks)))
    for cls in classes:
        amount_matching=len([pair for pair in classified_chunks
                             if pair[1]==cls])
        print("%s: total %d items" % (cls, amount_matching))


def run_with_config(config, train_set, test_set):
    print()
    print("Experiment: %s -> %s" %
          (estimator_name(config.transformer_config),
           estimator_name(config.classifier_config)))
    experiment=experiments.Experiment(config)
    result=experiment.run(train_set, test_set)
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
        print("Score time: %f seconds" % output.score_time)


def run_from_cli():
    parser=ArgumentParser()
    parser.add_argument("--algorithm", required=True,
                        choices=["metric", "features", "all", "ci"], )
    parser.add_argument("--transformer")
    parser.add_argument("--classifier")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--classes", required=True, nargs="+")
    parser.add_argument("--keep_borders", action="store_true")
    args=parser.parse_args()

    chunk_duration=pd.to_timedelta("%ds" % setup.chunk_duration_seconds)
    classified_chunks=get_classified_chunks(args.log_dir, args.classes, 
                                            chunk_duration, 
                                            not args.keep_borders)
    train_set, test_set=split_items_set(classified_chunks)
    display_chunks_stats(classified_chunks)
    
    configs=get_configs(args.algorithm, args.transformer, args.classifier)
    print("Configs to run:")
    print("\n".join(map(str, configs)))

    outputs=[run_with_config(config, train_set, test_set)
             for config in configs]
    results=zip(configs, outputs)
    display_results(results)


if __name__=="__main__":
    run_from_cli()

