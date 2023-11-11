import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


df = pd.read_csv("output.csv")

def train_test_val_split(dataset):
    X = df.drop(columns=['name_en', 'name_sa', 'unnamed: 0'])
    y = df['name_en']

    X_model, X_valid, y_model, y_valid = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    X_train, X_test, y_train, y_test = train_test_split(X_model,
                                                        y_model,
                                                        test_size=1 / 3,
                                                        random_state=10, stratify=y_model)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


X_train, X_test, X_valid, y_train, y_test, y_valid = train_test_val_split(df)


pipelines = {
    'rf': make_pipeline(RandomForestClassifier()),
    'gb': make_pipeline(GradientBoostingClassifier()),
}

# Fitting the models

def fitting_models(X_train, y_train):
    fit_models = {}
    for classifier, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[classifier] = model

    return fit_models


fit_models = fitting_models(X_train, y_train)


def evaluation_metrics(fit_models, X_test, y_test, X_valid, y_valid):
    for classifier in fit_models:
        print("----------------------------------------------")
        print("----------------------------------------------")
        print(f"evaluation of:", classifier)
        print("----------------------------------------------")
        print("----------------------------------------------")

        print("--------------")
        print("TRAIN-TEST")
        print("--------------")

        y_pred = classifier.predict(X_test)

        print('confusion matrix', classifier)
        print(confusion_matrix(y_test, y_pred))
        print('classification report')
        print(classification_report(y_test, y_pred))
        print('accuracy score')
        print(accuracy_score(y_test, y_pred))

        print("--------------")
        print("TRAIN-VALIDATION")
        print("--------------")

        y_pred = classifier.predict(X_valid)

        print('confusion matrix', classifier)
        print(confusion_matrix(y_valid, y_pred))
        print('classification report')
        print(classification_report(y_valid, y_pred))
        print('accuracy score')
        print(accuracy_score(y_valid, y_pred))


evaluation_metrics(fit_models, X_test, y_test, X_valid, y_valid)

best_classifier = fit_models[0]


def saving_fitted_model(best_classifier):
    filename = 'model.sav'
    pickle.dump(best_classifier, open(filename, 'wb'))
    print("-- model saved --")