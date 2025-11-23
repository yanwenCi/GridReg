import numpy as np
from scipy import stats 

# Assume you have the following data:
# voxelmorph
group1_mean = 5.79
group1_std = 4.12
group1_size = 38
# u-grid
group2_mean = 6.49
group2_std = 3.95
group2_size = 38
#keymorph
group3_mean = 0.91
group3_std = 0.0224
group3_size = 38
#transmorph
group4_mean = 0.89
group4_std = 0.0424
group4_size = 38
#u-grid-small
group5_mean = 0.89
group5_std = 0.0257
group5_size = 38

# Create the data arrays
group1 = np.random.normal(group1_mean, group1_std, group1_size)
group2 = np.random.normal(group2_mean, group2_std, group2_size)
group3 = np.random.normal(group3_mean, group3_std, group3_size)
group4 = np.random.normal(group4_mean, group4_std, group4_size)
group5 = np.random.normal(group5_mean, group5_std, group5_size)

# Perform the ANOVA test
f_stat, p_value = stats.f_oneway(group1, group2, group3, group4, group5)
# f_stat, p_value = stats.ttest_rel(group1, group2)

for group in [group1, group3, group4, group5]:
    f_stat, p_value = stats.ttest_ind(group, group2)
    print(" F-statistic:", f_stat, "p-value:", p_value)
# Interpret the results
alpha = 0.05  # Significance level
if p_value < alpha:
    print("There is a statistically significant difference between the means of the groups. p-value: ", p_value)
else:
    print("There is no statistically significant difference between the means of the groups. p-value: ", p_value)