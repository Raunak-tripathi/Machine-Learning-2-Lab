!pip install pgmpy
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define Bayesian Network structure
model = DiscreteBayesianNetwork([('T', 'S'), ('S', 'F')])

# Define Conditional Probability Distributions (CPDs)
cpd_t = TabularCPD(
    variable='T', variable_card=2,
    values=[[0.7], [0.3]],
    state_names={'T': ['Normal', 'High']}
)

cpd_s = TabularCPD(
    variable='S', variable_card=2,
    values=[[0.9, 0.4], [0.1, 0.6]],
    evidence=['T'], evidence_card=[2],
    state_names={'S': ['No', 'Yes'], 'T': ['Normal', 'High']}
)

cpd_f = TabularCPD(
    variable='F', variable_card=2,
    values=[[0.8, 0.2], [0.2, 0.8]],
    evidence=['S'], evidence_card=[2],
    state_names={'F': ['No', 'Yes'], 'S': ['No', 'Yes']}
)

# Add CPDs to model and verify
model.add_cpds(cpd_t, cpd_s, cpd_f)
model.check_model()

# Perform inference
inference = VariableElimination(model)
query_result = inference.query(variables=['F'], evidence={'T': 'High'})

print("Probability of Fever given High Temp:")
print(query_result)
