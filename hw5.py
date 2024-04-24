from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import numpy as np
import matplotlib as plt

def build_df(dir:str)->pd.DataFrame:
    """
    Builds a single dataframe out of all test subject csv data.
    Appends column for depression label, gender label, and speaker ID.

    Params:
    dir (str): name of directory to search through (features_test for test data, features_train for train)

    Return:
    dataframe with null values removed.
    """

    # grab feature names
    features = pd.read_csv('feature_description.csv',
                        encoding = 'ISO-8859-1', 
                        names=['feature', 'description'])['feature'].values.tolist()
    # depression and gender labels
    labels = pd.read_csv('labels.csv')

    # iterate through subject data
    df_list = []
    for filename in os.listdir(dir):
        if filename.split('.')[1] != 'csv':
            continue
        # speaker id
        speaker = filename.split('.')[0].split('_')[1]
        # convert to dataframe
        speaker_df = pd.read_csv(os.path.join(dir, filename), 
                                names=features)
        speaker_df.insert(len(speaker_df.columns), "speaker_id", [float(speaker)]*len(speaker_df), True)
        # append depression and gender labels to data
        depression = int(labels.loc[labels['Participant_ID']== float(speaker)]["Depression"].values[0])
        speaker_df.insert(len(speaker_df.columns), "depression", [depression]*len(speaker_df), True)
        gender = int(labels.loc[labels['Participant_ID']== float(speaker)]["Gender"].values[0])
        speaker_df.insert(len(speaker_df.columns), "gender", [gender]*len(speaker_df), True)
        df_list.append(speaker_df)
    # concatenate into single fataframe and remove nulls
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.replace(["NaN", '-999'], np.nan, inplace=True)
    combined_df.fillna(0, inplace=True)
    return combined_df


def analyze_results(start_df:pd.DataFrame, predicted_outcomes:np.array):
    """
    Concatenates all predictions on participants to a single value.
    Performs classification accuracy and balanced classification accuracy on entire dataset
        and on male and female participants, separately.
    Computes Equality of Opportunity.

    Params:
    start_df (pd.Dataframe): original dataframe with all features
    predicted_outcomes (np.array): prediction for each sample in start_df (must be the same size)

    Returns:
    class_acc, bca, male_class_acc, male_bca, female_class_acc, female_bca, eo
    """
    # average all predictions for participant
    start_df.insert(len(start_df.columns), 'prediction', predicted_outcomes)
    id_set = set(start_df['Participant_ID'])
    participant_arr = []
    for id in id_set:
        sub_df = start_df.loc[start_df['Participant_ID'] == id]
        avg_prediction = 1 if sub_df['prediction'].mean() > .5 else 0
        participant_arr.append([id, sub_df['gender'].values[0], sub_df['depression'].values[0], avg_prediction])
    test_df = pd.DataFrame(participant_arr, columns=['Participant_ID', 'gender', 'depression', 'prediction'])
    # compute accuracy and BCA over all participants
    class_acc = test_df[test_df['depression'] == test_df['prediction']].sum()/len(test_df)
    bca = (test_df[(test_df['depression'] == 0) & (test_df['prediction'] == 0)].sum()/len(test_df[test_df['depression'] == 0]) + 
            test_df[(test_df['depression'] == 1) & (test_df['prediction'] == 1)].sum()/len(test_df[test_df['depression'] == 1]))
    # compute accuracy and BCA for male participants
    male_df = test_df[test_df['gender'] == 1]
    male_class_acc = male_df[male_df['depression'] == male_df['prediction']].sum()/len(male_df)
    male_bca = (male_df[(male_df['depression'] == 0) & (male_df['prediction'] == 0)].sum()/len(male_df[male_df['depression'] == 0]) + 
            male_df[(male_df['depression'] == 1) & (male_df['prediction'] == 1)].sum()/len(male_df[male_df['depression'] == 1]))
    # compute accuracy and BCA for female participants
    female_df = test_df[test_df['gender'] == 0]
    female_class_acc = female_df[female_df['depression'] == female_df['prediction']].sum()/len(female_df)
    female_bca = (female_df[(female_df['depression'] == 0) & (female_df['prediction'] == 0)].sum()/len(female_df[female_df['depression'] == 0]) + 
            female_df[(female_df['depression'] == 1) & (female_df['prediction'] == 1)].sum()/len(female_df[female_df['depression'] == 1]))
    # compute equality of opportunity
    male_tpr = male_df[(male_df['depression'] == 1) & (male_df['prediction'] == 1)].sum()/len(male_df[male_df['depression'] == 1])
    female_tpr = female_df[(female_df['depression'] == 1) & (female_df['prediction'] == 1)].sum()/len(female_df[female_df['depression'] == 1])
    eo = 1 - abs(male_tpr - female_tpr)

    return class_acc, bca, male_class_acc, male_bca, female_class_acc, female_bca, eo

def train_random_forest(df, labels):
    x = 10

def depression_feature_selection(df:pd.DataFrame, test_df:pd.DataFrame):
    # perform depression classification on the data
    correlation_tups = []
    for col in df.columns:
        if col in ['Participant_ID', 'depression']:
            continue
        correlation_tups.append((col, df[col].corr(df['depression'])))
    correlation_tups = sorted(correlation_tups, key=lambda x: abs(x[1]), reverse=True)
    top_twenty_feats = [correlation_tups[x][0] for x in range(20)]

    # run model on filtered features
    labels = df['depression'].values.tolist()
    class_acc, bca, male_class_acc, male_bca, female_class_acc, female_bca, eo = [],[],[],[],[],[],[]
    for n in range(10, 51, 5):
        # select top performing features
        features = [correlation_tups[x][0] for x in range(n)]
        filtered_df = df.loc[:, features]
        # build random forest on these features
        best_rand_forest = train_random_forest(filtered_df, labels)
        predictions = best_rand_forest.predict(test_df.drop(columns=['depression']))
        # store accuracies
        results = analyze_results(test_df, predictions)
        class_acc.append(results[0])
        bca.append(results[1])
        male_class_acc.append(results[2])
        male_bca.append(results[3])
        female_class_acc.append(results[4])
        female_bca.append(results[5])
        eo.append(results[6])

    # plot results
    num_feats = range(10, 51, 5)
    plt.plot(num_feats, class_acc, label = "Classification Accuracy") 
    plt.plot(num_feats, male_class_acc, label = "Male Classification Accuracy") 
    plt.plot(num_feats, female_class_acc, label = "Female Classification Accuracy") 
    plt.plot(num_feats, bca, label = "Balanced Classification Accuracy") 
    plt.plot(num_feats, male_bca, label = "Male Balanced Classification Accuracy") 
    plt.plot(num_feats, female_bca, label = "Female Balanced Classification Accuracy") 
    plt.plot(num_feats, eo, label = "Equality of Opportunity")
    plt.legend() 
    # plt.show()

    return top_twenty_feats


