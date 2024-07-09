import warnings
warnings.filterwarnings('ignore')
PATH = "./data/"
import CauSumX
import pandas as pd
import Algorithms
import Utils
import csv
import json
import math

APRIORI = 0.65
COLUMNS_TO_EXCLUDE = ['education.num', 'capital.gain', 'capital.loss', 'income','fnlwgt']
COLUMNS_TO_EXCLUDE_WITHOUT_SALARY = ['education.num', 'capital.gain', 'capital.loss', 'income']
NUM_OF_FINAL_GROUPS = 3
TARGET_SALARY = 'income'
DAG = [
    'age;',
    'workclass;',
    'fnlwgt;',
    'education;',
    'education.num;',
    'marital.status;',
    'occupation;',
    'relationship;',
    'race;',
    'sex;',
    'capital.gain;',
    'capital.loss;',
    'hours.per.week;',
    'native.country;',
    'income;',
    'occupation_category;',
    # A
    'sex -> income;',
    'race -> income;',
    # C
    'age -> income;',
    'native.country -> income;',
    # M
    'sex -> marital.status;',
    'race -> marital.status;',
    'age -> marital.status;',
    'native.country -> marital.status;',
    'marital.status -> income;',
    # L
    'marital.status -> education.num;',
    'sex -> education.num;',
    'race -> education.num;',
    'age -> education.num;',
    'native.country -> education.num;',
    'education.num -> income;',
    # R
    'hours.per.week -> income;',
    'occupation -> income;',
    'workclass -> income;',
    'education.num -> hours.per.week;',
    'education.num -> occupation;',
    'education.num -> workclass;',
    'marital.status -> hours.per.week;',
    'marital.status -> occupation;',
    'marital.status -> workclass;',
    'age -> hours.per.week;',
    'native.country -> hours.per.week;',
    'age -> occupation;',
    'native.country -> occupation;',
    'age -> workclass;',
    'native.country -> workclass;',
    'sex -> hours.per.week;',
    'sex -> occupation;',
    'sex -> workclass;',
    'race -> hours.per.week;',
    'race -> occupation;',
    'race -> workclass;'
]

import pandas as pd
import csv

def process_group(df_origin, df, group, outcome_column, protected_column, protected_column_value, results):
    excluded_attrs = set(group.keys())
    excluded_attrs.update(["race", "sex", "age","relationship","marital.status"])
    remaining_attrs = [attr for attr in df.columns if attr not in excluded_attrs]
    
    treatments = Utils.getLevel1treatments(remaining_attrs, df)

    group_df = df_origin.copy()
    for k, v in group.items():
        group_df = group_df[group_df[k] == v]
    #CHECKING
    for k, v in group.items():
        group_df.drop(columns=k)
    #TO HERE
    group_size = len(group_df)
    num_females = len(group_df[group_df[protected_column] == protected_column_value])
    num_males = group_size - num_females
    dict_for_best_treatments =  dict()
    general_and_female_effectt=[]
    for treatment in treatments:
        best_Util=0
        df_origin['TempTreatment'] = df_origin.apply(lambda row: addTempTreatment(row, treatment, excluded_attrs), axis=1)
        group_df['TempTreatment'] = group_df.apply(lambda row: addTempTreatment(row, treatment, excluded_attrs), axis=1)

        treatment_effect = Utils.getTreatmentCATE(group_df, DAG, treatment, TARGET_SALARY)
        if(treatment_effect <=0):
            continue
        female_group_df = group_df[group_df[protected_column] == protected_column_value]
        female_treatment_effect = Utils.getTreatmentCATE(female_group_df, DAG, treatment, TARGET_SALARY)
        if math.isnan(female_treatment_effect):
            female_treatment_effect = 0
        utility_grade = 0.5 * (female_treatment_effect/treatment_effect) + 0.5*treatment_effect
        if(utility_grade>best_Util):
            best_Util = utility_grade
            general_and_female_effectt=[treatment_effect,female_treatment_effect]
            group_key = json.dumps(group)
            if group_key in dict_for_best_treatments:
                dict_for_best_treatments[group_key][1] = utility_grade
                dict_for_best_treatments[group_key][0] = treatment
            else:
                dict_for_best_treatments[group_key] = [treatment,utility_grade]
    result = {
            'group': group,
            'group_size': group_size,
            'num_females': num_females,
            'num_males': num_males,
            'treatment': dict_for_best_treatments[json.dumps(group)][0],
            'treatment_effect': general_and_female_effectt[0],
            'female_treatment_effect': general_and_female_effectt[1],
            'utility_grade': dict_for_best_treatments[json.dumps(group)][1]
        }
        
    results.append(result)

def so(outcome_column, protected_column, protected_column_value):
    df = pd.read_csv(PATH + 'adult_new.csv', encoding='utf8')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna()
    df_origin = df.copy()
    df = df.drop(columns=COLUMNS_TO_EXCLUDE)
    groups = Algorithms.getAllGroups(df, APRIORI)
    print('num of groups: ', len(groups))
    results = []

    for group in groups:
        process_group(df_origin, df, group, outcome_column, protected_column, protected_column_value, results)

    # Sort results by group name
    results = greedy_selection(df_origin, NUM_OF_FINAL_GROUPS, results)
    results.sort(key=lambda x: str(x['group']))

    keys = results[0].keys()
    with open('treatment_effects_greedy.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def addTempTreatment(row, treatment, ordinal_atts):
    for att, val in treatment.items():
        if att in ordinal_atts:
            if row[att] < val:
                return 0
            else:
                return 1
        else:
            if row[att] == val:
                return 1
    return 0

def getAttsVals(atts, df):
    atts_vals = {}
    for att in atts:
        atts_vals[att] = df[att].unique().tolist()
    return atts_vals

def greedy_selection(df, k, results):
        #result = {
        #    'group': group,
        #    'group_size': group_size,
        #    'num_females': num_females,
        #    'num_males': num_males,
        #    'treatment': dict_for_best_treatments[json.dumps(group)][0],
        #    'treatment_effect': general_and_female_effectt[0],
        #    'female_treatment_effect': general_and_female_effectt[1],
        #    'utility_grade': dict_for_best_treatments[json.dumps(group)][1]
        #}
    selected_result = []

    for _ in range(k):
        best_group = None
        best_result = None
        best_group_size = 0
        best_score = -1

        # Find the group with the highest score (group_size * utility_grade)
        for result in results:
            group = result['group']
            utility_grade = result['utility_grade']
            
            # Calculate the current size of the group in the DataFrame
            group_size = df[df.apply(lambda row: all(row[k] == v for k, v in group.items()), axis=1)].shape[0]
            
            score = group_size * utility_grade
            if score ==0:
                continue

            if score > best_score:
                best_score = score
                best_group = group
                best_result = result
                best_group_size = group_size

        if best_group is None:
            break
        
        # Add the best group to the selected groups
        best_result['score'] = best_score
        best_result['group'] = best_group
        best_result['group_size'] = best_group_size
        selected_result.append(best_result)

        # Get the DataFrame rows corresponding to the selected group
        group_df = df[df.apply(lambda row: all(row[k] == v for k, v in best_group.items()), axis=1)]
        
        # Remove the selected group rows from the DataFrame
        df = df.drop(group_df.index)

    return selected_result


def main():
   outcome_column = "fnlwgt"
   protected_column='sex'
   protected_value = 'Female'
   so(outcome_column, protected_column, protected_value)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
