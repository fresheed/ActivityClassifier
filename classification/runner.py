#! /usr/bin/python3
from classification.metric import knn
from sklearn.metrics import confusion_matrix, accuracy_score


def manual_dist(one, two):
    if len(one)!=2 or len(two)!=2:
        raise ValueError("len must be 2")
    manh=sum(abs(val1-val2) 
             for val1, val2 in zip(one, two))
    return manh


def main():
    points_1=(0, 0), (1, 0), (1, 1), (2, 0)
    points_2=(3, 3), (3, 4), (3, 5), (5, 5)
    points=points_1+points_2
    classes=[0, ]*4+[1, ]*4

    test_points=[(2, 2.5), (5, 2)]
    test_classes=[0, 1]

    for classificator in [knn.KNNClassifier]:
        print("\nUsing %s" % classificator.__name__)
        classifier=classificator(manual_dist)
        trained_model=classifier.train(points, classes)
        classified=trained_model.classify(test_points)
        confmat=confusion_matrix(test_classes, classified)
        print("Confusion:")
        print(confmat)
        print("Accuracy:", accuracy_score(test_classes, classified))
        

if __name__=="__main__":
    main()
