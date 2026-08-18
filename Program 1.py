#Program 1

import pandas as pd

data=pd.DataFrame([
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
    ],
    columns=['Sky','AirTemp','Humidity','Wind','Water','Forecast','EnjoySport'])

print("Dataset:\n",data)

attributes = data.columns[:-1]
target = data.columns[-1]

def find_s(df):
    hypothesis = ['0']*(len(df.columns)-1)
    for i,row in df.iterrows():
        if row[target] == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j]=='0':
                    hypothesis[j]=row[j]
                elif hypothesis[j]!=row[j]:
                    hypothesis[j]='?'
    return hypothesis

final_hypothesis=find_s(data)
print("\nMost specific hypothesis (Find-S):",final_hypothesis)


def can_elimination(df):
    S = [['0']*(len(df.columns)-1)]
    G = [['?']*(len(df.columns)-1)]

    for i,row in df.iterrows():
        if row[target]=="Yes":
            G=[g for g in G if all(g[j]=='?'or g[j]==row[j] for j in range(len(attributes)))]
            for j in range(len(attributes)):
                if S[0][j]=='0':
                    S[0][j]=row[j]
                elif S[0][j]!=row[j]:
                    S[0][j]='?'
        else:
            S=[s for s in S if not all(s[j]=='?' or s[j]==row[j] for j in range(len(attributes)))]
            new_G=[]
            for g in G:
                for j in range(len(attributes)):
                    if g[j]=='?':
                        for val in df[attributes[j]].unique():
                            if val!=row[j]:
                                new_hypo=g.copy()
                                new_hypo[j]=val
                                if any(all(s[k]=='?'or s[k]==new_hypo[k] or s[k]=='0' for k in range(len(attributes)))for s in S):
                                    new_G.append(new_hypo)
            G=new_G
    return S, G

S_final, G_final = can_elimination(data)
print("\n Candidate Elimination Results:")
print("S (Most specific boundary):",S_final)
print("G (Most general boundary):",G_final)
