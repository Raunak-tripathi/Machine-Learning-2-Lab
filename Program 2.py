!pip install wittgenstein pandas scikit-learn

import pandas as pd
from sklearn.datasets import load_iris
import wittgenstein as lw

iris = load_iris(as_frame=True)
x=iris.data
y=iris.target

# print(iris.feature_names)
# print(iris.target_names)
# print(iris.target)

df = pd.concat([x,y.rename("target")],axis=1)
print("Dataset head:\n", df.head())

model = lw.RIPPER()
model.fit(x,y, pos_class=0)
print("\n===RIPPER Rules ===")
print(model.ruleset_)

df_bin = df.copy()
df_bin['target'] = (df_bin['target'] == 0).astype(int)

attributes = list(x.columns)

def foil_gain(pos_before,neg_before,pos_after,neg_after):
    if pos_after==0:
        return -1e9
    return pos_after * (((pos_after)/(pos_after+neg_after))-((pos_before)/(pos_before+neg_before)))

def foil(df,target_col='target'):
    rules=[]
    pos_total = df[target_col].sum()
    neg_total = len(df)-pos_total

    while pos_total > 0:
        rule=[]
        pos_rem,neg_rem = pos_total, neg_total
        covered = df.copy()

        while neg_rem >0:
            best_gain, best_attr = -1e9, None
            for attr in attributes:
                for val in df[attr].unique():
                    subset = covered[covered[attr]==val]
                    pos_after = subset[target_col].sum()
                    neg_after = len(subset)-pos_after
                    gain = foil_gain(pos_rem,neg_rem,pos_after,neg_after)
                    if gain > best_gain:
                        best_gain, best_attr, best_val, best_subset = gain,attr,val,subset
            if best_attr is None:
                break
            rule.append((best_attr,best_val))
            covered = best_subset
            pos_rem, neg_rem = covered[target_col].sum(),len(covered)-covered[target_col].sum()

        rules.append(rule)
        df = df.drop(covered.index)
        pos_total, neg_total = df[target_col].sum(), len(df)-df[target_col].sum()
    return rules

rules=foil(df_bin)
print("\n=== FOIL Learned Rules (Setosa vs Not) ===")
for conds in rules:
    cond_str = " AND ".join([f"{a}={v}" for a,v in conds])
    print(f"IF {cond_str} THEN class=0")
