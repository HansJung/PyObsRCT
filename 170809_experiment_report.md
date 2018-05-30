Experiment 1. ARDSNet (2000)

Date: 170810 14:00

Validation 

- Severe ARDS <= 150
- train:test = 0.75:0.25 
- validation result: 0.68

Variables 

- Control 

  ```python
  # Intervention
  VT_val = 7 # 12 if high VT 
  FO2_allowed = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ])
  PEEP_allowed = np.array([5,8,10,10,12,14,16,18])
  min_PEEP = 5
  maxmax_PEEP = 24
  maxmin_PEEP = 18
  min_RR = 6
  max_RR = 35
  PP_limit = 30 # 50 if high PP 
  ```

  - Specify VT_value 
  - Allow a suggested FiO2, PEEP combination 
  - RR above 6, less than 35 
  - PP less than 30.



Code

```python
for i in range(N):
    ''' For each individual'''
    patients_info = X.iloc[[i]]
    patients_idx = X.iloc[[i]].index[0]
    # Intervention to VT
    intervened_info = patients_info.set_value(patients_idx, 'VT_WEIGHT', VT_val)

    # Intervention to FO2 and PEEP
    curr_FO2 = round(intervened_info.get_value(patients_idx, 'FiO2'),1)
    curr_PEEP = patients_info.get_value(patients_idx, 'PEEP')
    if (np.isnan(curr_FO2) == False and curr_FO2 < 0.3):
        intv_FO2 = 0.3
        intv_PEEP = max(5, curr_PEEP)
    elif(np.isnan(curr_FO2) == False and curr_FO2 > 0.9):
        intv_PEEP = min(maxmax_PEEP, curr_PEEP)
        intv_PEEP = max(maxmin_PEEP, intv_PEEP)
    else:
        intv_PEEP = PEEP_allowed[np.where(FO2_allowed == 0.5)][0]
    intervened_info = intervened_info.set_value(patients_idx, 'PEEP', intv_PEEP)

    # Intervention to RR
    curr_RR = intervened_info.get_value(patients_idx,'ResRate')
    intv_RR = max(min_RR, curr_RR)
    intv_RR = min(max_RR, intv_RR)
    intervened_info = intervened_info.set_value(patients_idx, 'ResRate', intv_RR)

    # Intervention to PP
    curr_PP = intervened_info.get_value(patients_idx, 'PP')
    intv_PP = min(curr_PP, PP_limit)
    intervened_info = intervened_info.set_value(patients_idx, 'PP', intv_RR)

    ''' Control confounder variable '''
    sum_adj = 0
    for j in range(N):
        adj_Berlin = Berlin.iloc[j]
        adj_Age = Age.iloc[j]
        adj_Sex = Sex.iloc[j]
        adj_Weight = Weight.iloc[j]

        # put the value to compute P(Y | I, M, C)
        intervened_info = intervened_info.set_value(patients_idx,'Sev',adj_Berlin)
        intervened_info = intervened_info.set_value(patients_idx, 'AGE', adj_Age)
        intervened_info = intervened_info.set_value(patients_idx, 'WEIGHT', adj_Weight)
        intervened_info = intervened_info.set_value(patients_idx, 'GENDER', adj_Sex)

        sum_adj += loaded_model.predict_proba(intervened_info)[0][1] # prob of survival
    sum_adj /= N
    survival_prob.append(sum_adj)
    print(i, round(i/N,4), round(sum_adj,2))
```

Result 

| Intervention | Survival rate |
| ------------ | ------------- |
| Low VT       | 0.416         |
| High VT      | 0.365         |







------

Experiment 2. ARDSNet (2004) - High PEEP 

Date: 170809 14:30 

Validation 

- Severe ARDS <= 150
- train:test = 0.75:0.25 
- validation result: 0.68



Intervened Variables 

```python
VT_val = 7 
# VT_val = 12 
FO2_allowed = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
PEEP_allowed = np.array([12, 14, 16, 16, 20, 20, 22, 22]) # High PEEP 
min_PEEP = 12
maxmax_PEEP = 24
maxmin_PEEP = 22
min_RR = 6
max_RR = 35
PP_limit = 30
```

Result

| Intervention            | Survival rate |
| ----------------------- | ------------- |
| Low PEEP (with low VT)  | 0.416         |
| High PEEP (with low VT) | 0.415         |



----

Experiment 3. Papazian (2010) - NMBA treatment 

Date: 170809 15:40 

Validation 

- Severe ARDS <= 150
- train:test = 0.75:0.25 
- validation result: 0.68 

Intervened Variables 

```python
VT_val = 7
FO2_allowed = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
PEEP_allowed = np.array([12, 14, 16, 16, 20, 20, 22, 22])
min_PEEP = 12
maxmax_PEEP = 24
maxmin_PEEP = 22
min_RR = 6
max_RR = 35
PP_limit = 30
interv_NMBA = 100
```

Code

```python
# Intervention to NMBA
intervened_info = intervened_info.set_value(patients_idx,'NMBA',interv_NMBA)
```

Result

* 0.413

---

Experiment 4. Papazian (2010) - NMBA treatment / low PEEP 

Date: 170809 16:40 

Validation 

- Severe ARDS <= 150
- train:test = 0.75:0.25 
- validation result: 0.68 

Intervened Variables 

```python
VT_val = 7
FO2_allowed = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
PEEP_allowed = np.array([5,8,10,10,12,14,16, 18])
min_PEEP = 5
maxmax_PEEP = 24
maxmin_PEEP = 18
min_RR = 6
max_RR = 35
PP_limit = 30
interv_NMBA = 100
```

Code

```python
# Intervention to NMBA
if Papazian == True:
	intervened_info = intervened_info.set_value(patients_idx,'NMBA',interv_NMBA)
```

Result

| Intervention             | Survival rate |
| ------------------------ | ------------- |
| NMBA (100) with low PEEP | 0.410         |
| No NMBA with low PEEP    | 0.418         |
| NMBA (150) with low PEEP | 0.315         |



---

Report 

| RCT                   | Intervent   | Result |
| --------------------- | ----------- | ------ |
| RCT 1 ARDSNet (2000)  | VT = low    | 0.416  |
|                       | VT = high   | 0.365  |
| RCT 2 ARDSNet (2004)  | PEEP = low  | 0.416  |
|                       | PEEP = high | 0.415  |
| RCT 3 Papazian (2010) | NMBA = YES  | 0.410  |
|                       | NMBA = No   | 0.418  |

