import warnings
import pandas as pd
from itertools import combinations
import heapq

warnings.filterwarnings('ignore')
PATH = "./data/adult_new.csv"
THRESHOLD = 0.3
COLUMNS_TO_EXCLUDE = ['education.num', 'capital.gain', 'capital.loss', 'income','fnlwgt','age']
MINIMUM_SIZE_OF_ITEMSET = 2
MAXIMUM_SIZE_OF_ITEMSET = 5
TOP_X_RESULTS = 10
DESIRED_COVERAGE_PERCENTAGE = 0.95

def generate_candidates(prev_candidates, k):
    candidates = []
    len_prev = len(prev_candidates)
    for i in range(len_prev):
        for j in range(i + 1, len_prev):
            candidate = sorted(list(set(prev_candidates[i]) | set(prev_candidates[j])))
            if len(candidate) == k:
                if all(sorted(subset) in prev_candidates for subset in combinations(candidate, k - 1)):
                    candidates.append(candidate)
    return candidates

def support_count(df, candidate):
    mask = df[list(candidate)].all(axis=1)
    return mask.sum(), mask

def apriori(df):
    initial_itemsets = [[col] for col in df.columns if col not in COLUMNS_TO_EXCLUDE]
    frequent_itemsets = []
    current_frequent_itemsets = []
    covered_transactions = set()

    # First pass: process single-item itemsets
    for itemset in initial_itemsets:
        support, transaction_mask = support_count(df, itemset)
        if support >= THRESHOLD * len(df):
            current_frequent_itemsets.append((support, itemset, transaction_mask))

    # Generate frequent itemsets of size 2 and above
    k = MINIMUM_SIZE_OF_ITEMSET
    while k <= MAXIMUM_SIZE_OF_ITEMSET:
        candidates = generate_candidates([itemset for _, itemset, _ in current_frequent_itemsets], k)
        current_frequent_itemsets = []
        for candidate in candidates:
            support, transaction_mask = support_count(df, candidate)
            if support >= THRESHOLD * len(df):
                current_frequent_itemsets.append((support, candidate, transaction_mask))
                # Only add itemsets of size greater than 1
                if len(candidate) > 1:
                    frequent_itemsets.append((support, candidate, transaction_mask))
        
        for _, _, transaction_mask in current_frequent_itemsets:
            covered_transactions.update(transaction_mask[transaction_mask].index)
        coverage = len(covered_transactions) / len(df)
        if coverage >= DESIRED_COVERAGE_PERCENTAGE:
            print(f"Desired coverage of {DESIRED_COVERAGE_PERCENTAGE*100}% achieved. Stopping early.")
            break
        
        k += 1

    return heapq.nlargest(TOP_X_RESULTS, frequent_itemsets, key=lambda x: x[0])

def collect_and_apriori():
    try:
        df = pd.read_csv(PATH, encoding='utf8')
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return
    
    df = df.drop(columns=COLUMNS_TO_EXCLUDE)

    frequent_itemsets = apriori(df)
    print("Top Frequent Itemsets:")
    for support, itemset, _ in frequent_itemsets:
        print(f"Support: {support}, Itemset: {itemset}")

def main():
    collect_and_apriori()

if __name__ == '__main__':
    main()
