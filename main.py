import warnings
warnings.filterwarnings('ignore')
PATH = "./data/"
import CauSumX
import pandas as pd
import Algorithms
import Utils
import csv
APRIORI = 0.65
COLUMNS_TO_EXCLUDE = ['education.num', 'capital.gain', 'capital.loss', 'income','fnlwgt']
COLUMNS_TO_EXCLUDE_WITHOUT_SALARY = ['education.num', 'capital.gain', 'capital.loss', 'income']




def so(outcome_column,protected_column,protected_column_value):
    df = pd.read_csv(PATH + 'adult_new.csv', encoding='utf8')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df_origin = df
    df = df.drop(columns=COLUMNS_TO_EXCLUDE)
    df = df.dropna()
    columns = df.columns
    groups = Algorithms.getAllGroups(df, APRIORI)
    print('num of groups: ', len(groups))
    results = []
    for group in groups:
        excluded_attrs = set(group.keys())
        excluded_attrs.update(["race", "sex", "age"])
        remaining_attrs = [attr for attr in columns if attr not in excluded_attrs]
        
        treatments = Utils.getLevel1treatments(remaining_attrs, df, ordinal_atts=[])

        df = df_origin.drop(columns = COLUMNS_TO_EXCLUDE_WITHOUT_SALARY).dropna()
        group_df = df[df.apply(lambda row: all(row[k] == v for k, v in group.items()), axis=1)]
        group_size = len(group_df)
        num_females = len(group_df[group_df[protected_column] == protected_column_value])
        num_males = group_size - num_females
        female_df = df[df[protected_column] == protected_column_value]  
        for treatment in treatments:
            df['TempTreatment'] = df.apply(lambda row: addTempTreatment(row, treatment, excluded_attrs), axis=1)
            female_df['TempTreatment'] = female_df.apply(lambda row: addTempTreatment(row, treatment, excluded_attrs), axis=1)

            treated_outcome_mean = df[df['TempTreatment'] == 1][outcome_column].mean()
            control_outcome_mean = df[df['TempTreatment'] == 0][outcome_column].mean()
            
            treatment_effect = abs(treated_outcome_mean - control_outcome_mean)
             # Calculate treatment effect for females
            female_treated_outcome_mean = female_df[female_df['TempTreatment'] == 1][outcome_column].mean()
            female_control_outcome_mean = female_df[female_df['TempTreatment'] == 0][outcome_column].mean()
            
            female_treatment_effect = abs(female_treated_outcome_mean - female_control_outcome_mean)
            utility_grade = 0.5*(female_treatment_effect/treatment_effect) + 0.5*treatment_effect
            results.append({
                'group': group,
                'group_size': group_size,
                'num_females': num_females,
                'num_males': num_males,
                'treatment': treatment,
                'treatment_effect': treatment_effect,
                'female_treatment_effect': female_treatment_effect,
                'utility_grade': utility_grade
            })
    keys = results[0].keys()
    with open('treatment_effects.csv', 'w', newline='') as output_file:
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

def main():
   outcome_column = "fnlwgt"
   protected_column='sex'
   protected_value = 'Female'
   so(outcome_column,protected_column,protected_value)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
