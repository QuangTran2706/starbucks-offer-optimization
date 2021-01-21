from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, auc
import pickle
import pandas as pd
import numpy as np
import os

best_model = None
df = pd.read_csv("./data/new_data.csv", sep=',', decimal='.')
cleanData_df = pd.read_csv("./data/cleanData.csv", sep=',', decimal='.', low_memory=False)
portfolio = pd.read_csv("./data/portfolio_new.csv", sep=',', decimal='.')

def getData(dataFrame):
    X = dataFrame.drop(['customer_id', 'amount', 'event_offer completed', 'event_offer received', 'event_offer viewed', 'event_transaction', 'offer_succeed'], axis=1)
    Y = dataFrame['offer_succeed']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return [X_train, X_test, y_train, y_test]

def evaluate_model_performance(clf, X_train, y_train):  
    y_pred_rf = clf.predict(X_train)
    clf_accuracy = accuracy_score(y_train, y_pred_rf)
    clf_f1_score = f1_score(y_train, y_pred_rf)

    return clf_accuracy, clf_f1_score

def loadModels():
    adaClf_filename = "model/adaClf_model.pkl"
    logRegression_filename = "model/logRegression_model.pkl"
    lgbmClassifier = "model/lgbmClf_model.pkl"
    list_models_name = [adaClf_filename, logRegression_filename, lgbmClassifier]
    models = [{'name' : 'AdaBoostClassifier'}, {'name' : 'LogisticRegression'}, {'name' : 'LGBMClassifier'}]
    for n, i in enumerate(list_models_name):
        with open(i, 'rb') as file:
            models[n]['model'] = pickle.load(file)
    return models
    
def loadModel(model_name):
    if model_name == 'adaClf':
        filename = "model/adaClf_model.pkl"
    elif model_name == 'logReg':
        filename = "model/logRegression_model.pkl"
    elif model_name == 'lgbmClf':
        filename = "model/lgbmClf_model.pkl"
    else:
        filename = "model/lgbmClf_model.pkl"

    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def getModelScore(models, X_test, y_test):
    model_scores = []
    for item in models:
        model_name = item['name']
        acc , f1 = evaluate_model_performance(item['model'],  X_test, y_test)
        model_scores.append([model_name, acc, f1])
    return model_scores

def predict_offer_success(customer_id, offer_id, clf_model, time=0.0, amount_reward=0.0):
    """
    This function using model to predict whether the offer type apply to this type of customer is effective or not
    """
    # create customer profile
    customer_profile = df.drop(['offer_reward', 'difficulty', 'duration', 'offer id', 'channel_email', 'channel_mobile', 'channel_social', 'channel_web', 
                           'offer_type_bogo', 'offer_type_discount', 'offer_type_informational',  'amount', 'event_offer completed', 
                           'event_offer received', 'event_offer viewed', 'event_transaction', 'offer_succeed', 'time', 'amount_rewarded'], axis=1)
    customer_profile = customer_profile.drop_duplicates()
    # Create offer profile
    offers = portfolio.copy()
    offers.rename(columns={'reward': 'offer_reward',
                       'email': 'channel_email',
                       'mobile': 'channel_mobile',
                       'social': 'channel_social',
                       'web': 'channel_web'}, inplace=True)
    
    offer = offers[offers['offer id'] == offer_id].reset_index()
    cust = customer_profile[customer_profile['customer_id'] == customer_id].reset_index()
    
    # make featres
    feature = pd.concat([cust, offer], axis=1)
    feature.drop(["customer_id"], axis=1)
    feature['time'] = time
    feature['amount_rewarded'] = amount_reward
    # predict
    list_features = ['time', 'offer id', 'amount_rewarded', 'offer_reward', 'difficulty', 'duration',
                 'channel_email', 'channel_mobile', 'channel_social', 'channel_web', 'offer_type_bogo', 'offer_type_discount',
                'offer_type_informational', 'age', 'income', 'membership_duration', 'gender_F', 'gender_M', 'gender_O']
    success = clf_model.predict(feature[list_features])
    result = False
    if success[0] == 0:
        print('This type of offer is not effective against the current customer')
    else:
        print("offer is effective")
        result = True
    return result

def get_evaluation_report(x_test, y_test, model):
    y_pred = model.predict(x_test)
    return classification_report(y_test, y_pred, output_dict=True)

def get_predict_result(x_test, model):
    y_pred = model.predict(x_test)
    return y_pred

def get_roc_curve(model, x, y_true):
    y_prob = model.predict_proba(x)[::,1]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label =True)
    sum_ss = tpr + (1-fpr) # sum of sum_sensitivity and specificity
    best_threshold_id = np.argmax(sum_ss)
    best_fpr = fpr[best_threshold_id]
    best_tpr = tpr[best_threshold_id]
    auc_train = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_train, best_fpr, best_tpr
