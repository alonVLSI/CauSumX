import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import Algorithms
import Utils
import csv
import json
import math
import time
import pandas as pd
import csv
from itertools import combinations

APRIORI = 0.6
PATH = "./data/"
COLUMNS_TO_EXCLUDE = ['education.num', 'capital.gain', 'capital.loss', 'income','fnlwgt']
COLUMNS_TO_EXCLUDE_WITHOUT_SALARY = ['education.num', 'capital.gain', 'capital.loss', 'income']
NUM_OF_FINAL_GROUPS = 7
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

def process_group(df_origin, df, group, protected_column, protected_column_value, results):
    excluded_attrs = set(group.keys())
    excluded_attrs.update(["race", "sex", "age","relationship","marital.status"])
    remaining_attrs = [attr for attr in df.columns if attr not in excluded_attrs]
    
    treatments = Utils.getLevel1treatments(remaining_attrs, df)
    group_df = df_origin.copy()
    for k, v in group.items():
        group_df = group_df[group_df[k] == v]
    group_size = len(group_df)

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
            'males':0,
            'females':0,
            'treatment': dict_for_best_treatments[json.dumps(group)][0],
            'treatment_effect': general_and_female_effectt[0],
            'female_treatment_effect': general_and_female_effectt[1],
            'utility_grade': dict_for_best_treatments[json.dumps(group)][1]
        }
        
    results.append(result)

def CalculateEqualyTreatments(protected_column, protected_column_value):
    start_time = time.time()
    df = pd.read_csv(PATH + 'adult_new.csv', encoding='utf8')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna()
    df_origin = df.copy()
    df = df.drop(columns=COLUMNS_TO_EXCLUDE)
    total_size = len(df_origin)
    groups = Algorithms.getAllGroups(df, APRIORI)
    results = []

    for group in groups:
        process_group(df_origin, df, group, protected_column, protected_column_value, results)

    results = greedy_selection(df_origin, NUM_OF_FINAL_GROUPS, results,protected_column, protected_column_value)
    results.sort(key=lambda x: str(x['group']))

    expectation_grade = calculateExpectation(results,df_origin)
    protected_expectation_grade = calculateExpectation(results,df_origin,{protected_column:protected_column_value})
    end_time = time.time()
    overall_time = end_time-start_time
    total_coverage = sum(item['group_size'] for item in results)
    summary = {
        'expectation_grade': expectation_grade,
        'protected_expectation_grade': protected_expectation_grade,
        'Total time(seconds)': overall_time,
        'Total amount': total_size,
        'Total Coverage': total_coverage,
        'Coverage Percentae': f"{((total_size/total_coverage)*100):.2f}%"
    }

    with open(f'treatment_effects_greedy_{NUM_OF_FINAL_GROUPS}_final_groups_{APRIORI}AP.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        if results:
            writer.writerow(results[0].keys())
        for item in results:
            writer.writerow(item.values())
        writer.writerow([])
        if summary:
            writer.writerow(summary.keys())  
            writer.writerow(summary.values())

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

def greedy_selection(df, k, results,protected_column,protected_column_value):
    selected_result = []
    for _ in range(k):
        best_group = None
        best_result = None
        best_group_size = 0
        best_score = -1
        num_females = 0
        num_males = 0
        # Find the group with the highest score (group_size * utility_grade)
        for result in results:
            group = result['group']
            utility_grade = result['utility_grade']
            
            # Calculate the current size of the group in the DataFrame
            filtered_df = df[df.apply(lambda row: all(row[k] == v for k, v in group.items()), axis=1)]
            group_size = len(filtered_df)
            
            score = group_size * utility_grade
            if score ==0:
                continue

            if score > best_score:
                best_score = score
                best_group = group
                best_result = result
                best_group_size = group_size
                num_females = len(filtered_df[filtered_df[protected_column] == protected_column_value])
                num_males = group_size - num_females
        if best_group is None:
            break
        
        best_result['group'] = best_group
        best_result['group_size'] = best_group_size
        best_result['males'] = num_males
        best_result['females'] = num_females
        selected_result.append(best_result)

        group_df = df[df.apply(lambda row: all(row[k] == v for k, v in best_group.items()), axis=1)]
        
        df = df.drop(group_df.index)

    return selected_result

def calculateExpectation(results, df,protected=None):
    """
    Calculate the expected grade for the provided groups based on their CATE and sizes.   
    :param results: List of dictionaries, each containing group information.
    :param df: DataFrame containing the data.
    :return: Expected grade.
    """
    
    def get_combined_group_size(group_combination):
        combined_group = {}
        if protected:
            combined_group.update(protected)
        for group in group_combination:
            combined_group.update(group['group'])
        
        group_df = df[df.apply(lambda row: all(row.get(k) == v for k, v in combined_group.items()), axis=1)]
        size = len(group_df)
        return size
    
    total_weighted_sum = 0
    total_size = 0
    num_groups = len(results)
    
    for i in range(1, num_groups + 1):
        for group_combination in combinations(results, i):
            size = get_combined_group_size(group_combination)
            if size == 0:
                continue
            if protected:
                treatment_effects = [group['female_treatment_effect'] for group in group_combination]
            else:
                treatment_effects = [group['treatment_effect'] for group in group_combination] 
            min_treatment_effect = min(treatment_effects)
            weighted_sum = min_treatment_effect * size
            total_weighted_sum += weighted_sum
            total_size += size
    
    expected_grade = total_weighted_sum / total_size if total_size > 0 else 0
    return expected_grade

def main():
   protected_column='sex'
   protected_value = 'Female'
   CalculateEqualyTreatments(protected_column, protected_value)

if __name__ == '__main__':
   main()
