```python
import pandas as pd
import numpy as np
import copy
```


```python
def Purity(probs, type_of_purity = "Entropy"):
    if type_of_purity == "Entropy":
        return(-1 * probs * np.log2(probs)).sum()
    elif type_of_purity == "Gini_Index":
        return(1- (probs**2).sum())
    elif type_of_purity == "Majority_Error": 
        return(1- max(probs))
```


```python
def gain_info(data, attribute_column_name, label_column_name, type_of_gain = "Entropy"):
    weighted_purity = {} 
    Expected_Purity = 0
    Purity_1 =Purity(data[label_column_name].value_counts()/ len(data[label_column_name]), type_of_gain)
                                                    # calculates Entropy(S) by componentwise division
    group_names = data[attribute_column_name].unique()
    grouped_data = data[[attribute_column_name, label_column_name]].groupby(attribute_column_name)
    for name in group_names:
        #print ('group %s in attribute %s' %(name, attribute_column))
        X = grouped_data.get_group(name)[label_column_name]
        probs = X.value_counts()/ len(X)
        weight = len(X) / len(data[label_column_name])
        a = Purity(probs, type_of_gain)
        weighted_purity[name] = [weight, a]
        Expected_Purity += weight * a 
    return(weighted_purity,  Expected_Purity, Purity_1 - Expected_Purity)
```


```python
def build_tree(data, label_column_name, attributes = None, 
               values_of_attributes= None, depth= None, type_tree = "Entropy"):    
    
        #check if the list of attributes is given!  
    
    if attributes == None: 
        
            # create a dictionary containing attributes (names of columns in dataframe) 
            # and their corresponding values 
    
        list_of_attributes = list(data.keys())  # keys of dictionary
                # output is: ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
        list_of_attributes.remove(label_column_name) # remove the target column name
                # output is: ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        values = [list(data[key].unique()) for key in list_of_attributes] # gives values of 
                     # each attribute: ouptput is: 
                        # [['low', 'vhigh', 'high', 'med'],
                        # ['vhigh', 'high', 'med', 'low'],
                        # ['4', '5more', '2', '3'],
                        # ['4', '2', 'more'],
                        # ['big', 'med', 'small'],
                        # ['med', 'high', 'low']]
        values_of_attributes = dict(zip(list_of_attributes, values)) 
        # makes a dictionary with key = attributes and other= values  
        # output is: 
        # {'buying': ['low', 'vhigh', 'high', 'med'],
        #'maint': ['vhigh', 'high', 'med', 'low'],
        #'doors': ['4', '5more', '2', '3'],
        #'persons': ['4', '2', 'more'],
        #'lug_boot': ['big', 'med', 'small'],
        #'safety': ['med', 'high', 'low']}
        
    else:
        list_of_attributes = attributes.copy()
        
     # -----------------------------------------------------
    
    if len(list_of_attributes) == 0 or len(data[label_column_name].unique()) == 1 or depth == 0:  
        
        # return np.argmax(data[label_column_name].value_counts()) idxmax(axis = 0) 
        return data[label_column_name].value_counts().idxmax(axis = 0) 
                    # gives a label with majority (maximum number)
    
    else:
            # computing the Entropy of target column
    
        probs = data[label_column_name].value_counts() / len(data[label_column_name])
                # gives probabilities of labels
                # output for original label column is:
                # unacc    0.698
                # acc      0.222
                # good     0.045
                # vgood    0.035
        
        Target_Purity = Purity(probs, type_tree) # is a number: Entropy or Ginin index or Majority error
    
            # find the maximum gain information
    
        list_of_gains = [] # list of gains correspoinding to each attributes
    
        for attribute in list_of_attributes:
            list_of_gains.append(gain_info(data, attribute, label_column_name, type_tree)[2])
                # computes gain (according to type_tree) of each attribute (heads of columns)
                # so, output is a list of size 6 like:
                    # [0.10152470712485662, 0.07741985577459642, 0.006726514230977809,
                    #     0.22441128678577127, 0.03688725199484155, 0.25822501448993573]
    
        attribute_for_split = list_of_attributes[np.argmax(list_of_gains)]
        #attribute_for_split = list_of_attributes[list_of_gains.argmax()]
                # gives an attribute with maximum gain (e.g. for first time gives "safety")
        
        #print('Best attribute for split is', attribute_for_split)
    
               # constructing the tree for the current branch
        #print("stage 1: attribute_for_split = ", attribute_for_split)
    
        tree = {attribute_for_split:{}} 
                # "attribute_for_split" like "safety" is a node of tree
    
        #for value, group in data.groupby(attribute_for_split):
        grouped_data = data.groupby(attribute_for_split)
        list_of_attributes.remove(attribute_for_split)
        
        
        values_in_attribute_for_split = values_of_attributes[attribute_for_split]
        #values_of_attributes.pop(attribute_for_split, None)
        #print("stage 2: values_in_attribute_for_split =", values_in_attribute_for_split)
        
        
        for value in values_in_attribute_for_split: 
            
            #print(values_of_attributes)
            #print('Current value =', value)
        
            #splited_data.drop(columns = [attributes_for_split], inplace = True)
            #new_data = group.drop(columns = [attribute_for_split])
                 
            #print('this is values_in_attribute_for_split', values_in_attribute_for_split)
            # this line (above) removes attribute_for_split form values_of_attributes and 
            # returns its value. When attribute_for_split is not a key, it returns None. 
            
            #if value in values_in_attribute_for_split:
            #print('values in', attribute_for_split, '=', list(data[attribute_for_split].unique()))
            
            if value in list(data[attribute_for_split].unique()):
                #print("values_of_attributes=", values_in_attribute_for_split)
                new_data = grouped_data.get_group(value).drop(columns = [attribute_for_split])
                
                #print(new_data.head())
                       
            #print('new data is \n', new_data) 
            
                if depth!= None:
                    left_depth = depth -1
                    tt = type_tree 
                    subtree = build_tree(new_data, label_column_name, 
                            list_of_attributes, values_of_attributes, depth = left_depth, type_tree = tt) 
                else:
                    tt = type_tree
                    subtree = build_tree(new_data, label_column_name, 
                                     list_of_attributes, values_of_attributes, type_tree = tt)
        
                tree[attribute_for_split][value] = subtree
            else:
                tree[attribute_for_split][value] = data[label_column_name].value_counts().idxmax(axis = 0)
        
    return (tree)  
```


```python
def prediction(instance, trained_tree):
    #print(trained_tree.keys())
    #root = list(trained_tree.keys())[0]
    root = next(iter(trained_tree))
    #print('root is', root)
    if isinstance(trained_tree[root], dict):
        branch = instance[root]
        F = trained_tree[root][branch]
        if isinstance(F, dict):
            #return predict(trained_tree[root][branch], instance.drop(columns = [root])
            return prediction(instance, F)
        else:
            return trained_tree[root][branch]
```


```python
# stay
```


```python
A = open('data-desc.txt', 'r')
#for line in A.readlines():
    #print(line)
B = A.read()
print(B)
```

    1. Title: Bank Marketing
    
    2. Relevant Information:
    
       The data is related with direct marketing campaigns of a Portuguese banking institution. 
       The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
       in order to access if the product (bank term deposit) would be (or not) subscribed. 
    
       The classification goal is to predict if the client will subscribe a term deposit (variable y).
    
    3. Number of Attributes: 16 + output attribute.
    
    4. Attribute information:
    
       Input variables:
       # bank client data:
       1 - age (numeric)
       2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                           "blue-collar","self-employed","retired","technician","services") 
       3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
       4 - education (categorical: "unknown","secondary","primary","tertiary")
       5 - default: has credit in default? (binary: "yes","no")
       6 - balance: average yearly balance, in euros (numeric) 
       7 - housing: has housing loan? (binary: "yes","no")
       8 - loan: has personal loan? (binary: "yes","no")
       # related with the last contact of the current campaign:
       9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
      10 - day: last contact day of the month (numeric)
      11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
      12 - duration: last contact duration, in seconds (numeric)
       # other attributes:
      13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
      14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
      15 - previous: number of contacts performed before this campaign and for this client (numeric)
      16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
    
      Output variable (desired target):
      17 - y - has the client subscribed a term deposit? (binary: "yes","no")
    



```python
Columns_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
type_of_Attributes = ['numeric', 'categorical', 'categorical', 'categorical', 'binary', 'numeric', 
                      'binary', 'binary', 'categorical', 'numeric', 'categorical', 'numeric', 
                      'numeric', 'numeric', 'numeric', 'categorical', 'binary']
dic= dict(zip(Columns_names, type_of_Attributes))
```


```python
df_train = pd.read_csv('bank-train.csv', names = Columns_names)
df_test = pd.read_csv('bank-test.csv', names = Columns_names)
#df.head(5)
```


```python
median_dict = {}
df_train_new =pd.DataFrame()
df_test_new =pd.DataFrame()
for name in Columns_names:
    if dic[name] == 'numeric':
        M = df_train[name].median()
        median_dict[name] = M
        df_train_new[name+ '>' + str(M)] = np.where(df_train[name]  > M, "yes", 'no')
        df_test_new[name+ '>' + str(M)] = np.where(df_test[name]  > M, "yes", 'no')
    else:
        df_train_new[name] = df_train[name]
        df_test_new[name] = df_test[name]
```


```python
df_train_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age&gt;38.0</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance&gt;452.5</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day&gt;16.0</th>
      <th>month</th>
      <th>duration&gt;180.0</th>
      <th>campaign&gt;2.0</th>
      <th>pdays&gt;-1.0</th>
      <th>previous&gt;0.0</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>yes</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
      <td>may</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>1</td>
      <td>yes</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>no</td>
      <td>feb</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>2</td>
      <td>yes</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>cellular</td>
      <td>yes</td>
      <td>aug</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>success</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>3</td>
      <td>yes</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>no</td>
      <td>jul</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>4</td>
      <td>no</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>no</td>
      <td>apr</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age&gt;38.0</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance&gt;452.5</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day&gt;16.0</th>
      <th>month</th>
      <th>duration&gt;180.0</th>
      <th>campaign&gt;2.0</th>
      <th>pdays&gt;-1.0</th>
      <th>previous&gt;0.0</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>yes</td>
      <td>management</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>no</td>
      <td>jun</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>1</td>
      <td>yes</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>no</td>
      <td>may</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <td>2</td>
      <td>yes</td>
      <td>retired</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>yes</td>
      <td>jul</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>3</td>
      <td>no</td>
      <td>entrepreneur</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>no</td>
      <td>jun</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>4</td>
      <td>no</td>
      <td>student</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>yes</td>
      <td>jan</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
trained_tree = build_tree(df_train_new, 'y', depth= None, type_tree = "Entropy")
```


```python
from pprint import pprint
pprint(trained_tree) 
```

    {'duration>180.0': {'no': {'month': {'apr': {'job': {'admin.': {'education': {'primary': 'no',
                                                                                  'secondary': 'no',
                                                                                  'tertiary': {'housing': {'no': 'yes',
                                                                                                           'yes': 'no'}},
                                                                                  'unknown': 'no'}},
                                                         'blue-collar': {'poutcome': {'failure': 'no',
                                                                                      'other': 'no',
                                                                                      'success': 'yes',
                                                                                      'unknown': 'no'}},
                                                         'entrepreneur': 'no',
                                                         'housemaid': 'no',
                                                         'management': {'housing': {'no': {'balance>452.5': {'no': 'no',
                                                                                                             'yes': {'age>38.0': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                          'yes': 'no'}},
                                                                                                                                  'yes': 'no'}}}},
                                                                                    'yes': 'no'}},
                                                         'retired': 'no',
                                                         'self-employed': {'marital': {'divorced': 'yes',
                                                                                       'married': 'no',
                                                                                       'single': 'no'}},
                                                         'services': 'no',
                                                         'student': {'balance>452.5': {'no': 'yes',
                                                                                       'yes': 'no'}},
                                                         'technician': {'day>16.0': {'no': {'marital': {'divorced': 'yes',
                                                                                                        'married': {'balance>452.5': {'no': 'yes',
                                                                                                                                      'yes': {'housing': {'no': 'yes',
                                                                                                                                                          'yes': 'no'}}}},
                                                                                                        'single': 'no'}},
                                                                                     'yes': 'no'}},
                                                         'unemployed': 'no',
                                                         'unknown': 'no'}},
                                         'aug': {'poutcome': {'failure': {'job': {'admin.': 'no',
                                                                                  'blue-collar': 'no',
                                                                                  'entrepreneur': 'no',
                                                                                  'housemaid': 'no',
                                                                                  'management': {'marital': {'divorced': 'no',
                                                                                                             'married': 'yes',
                                                                                                             'single': {'age>38.0': {'no': {'balance>452.5': {'no': 'no',
                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                     'yes': 'yes'}}}},
                                                                                  'retired': 'no',
                                                                                  'self-employed': 'no',
                                                                                  'services': 'no',
                                                                                  'student': {'contact': {'cellular': 'yes',
                                                                                                          'telephone': 'no',
                                                                                                          'unknown': 'yes'}},
                                                                                  'technician': {'marital': {'divorced': 'no',
                                                                                                             'married': {'education': {'primary': 'yes',
                                                                                                                                       'secondary': 'yes',
                                                                                                                                       'tertiary': 'no',
                                                                                                                                       'unknown': 'yes'}},
                                                                                                             'single': 'no'}},
                                                                                  'unemployed': 'no',
                                                                                  'unknown': 'no'}},
                                                              'other': {'day>16.0': {'no': 'no',
                                                                                     'yes': 'yes'}},
                                                              'success': {'balance>452.5': {'no': 'yes',
                                                                                            'yes': 'no'}},
                                                              'unknown': {'job': {'admin.': 'no',
                                                                                  'blue-collar': 'no',
                                                                                  'entrepreneur': 'no',
                                                                                  'housemaid': 'no',
                                                                                  'management': 'no',
                                                                                  'retired': 'no',
                                                                                  'self-employed': 'no',
                                                                                  'services': 'no',
                                                                                  'student': 'no',
                                                                                  'technician': {'education': {'primary': 'no',
                                                                                                               'secondary': 'no',
                                                                                                               'tertiary': {'marital': {'divorced': 'no',
                                                                                                                                        'married': {'age>38.0': {'no': {'housing': {'no': {'balance>452.5': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                     'yes': {'default': {'no': {'loan': {'no': {'contact': {'cellular': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                      'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                             'yes': 'no'}}}},
                                                                                                                                                                                                                                                                                            'telephone': 'no',
                                                                                                                                                                                                                                                                                            'unknown': 'no'}},
                                                                                                                                                                                                                                                                         'yes': 'no'}},
                                                                                                                                                                                                                                                         'yes': 'no'}}}},
                                                                                                                                                                                                             'yes': {'campaign>2.0': {'no': {'default': {'no': {'loan': {'no': {'contact': {'cellular': {'day>16.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                            'telephone': 'yes',
                                                                                                                                                                                                                                                                                            'unknown': 'yes'}},
                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                      'yes': 'no'}}}},
                                                                                                                                                                                    'yes': 'no'}},
                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                        'single': 'no'}},
                                                                                                               'unknown': 'no'}},
                                                                                  'unemployed': {'balance>452.5': {'no': 'yes',
                                                                                                                   'yes': 'no'}},
                                                                                  'unknown': 'no'}}}},
                                         'dec': {'job': {'admin.': 'no',
                                                         'blue-collar': 'no',
                                                         'entrepreneur': 'no',
                                                         'housemaid': 'no',
                                                         'management': 'yes',
                                                         'retired': 'no',
                                                         'self-employed': 'no',
                                                         'services': 'no',
                                                         'student': 'no',
                                                         'technician': 'no',
                                                         'unemployed': 'no',
                                                         'unknown': 'no'}},
                                         'feb': {'job': {'admin.': 'no',
                                                         'blue-collar': {'poutcome': {'failure': 'no',
                                                                                      'other': 'no',
                                                                                      'success': 'yes',
                                                                                      'unknown': {'balance>452.5': {'no': {'age>38.0': {'no': 'no',
                                                                                                                                        'yes': {'education': {'primary': {'housing': {'no': 'no',
                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                              'secondary': 'no',
                                                                                                                                                              'tertiary': 'no',
                                                                                                                                                              'unknown': 'no'}}}},
                                                                                                                    'yes': 'no'}}}},
                                                         'entrepreneur': 'no',
                                                         'housemaid': 'no',
                                                         'management': {'education': {'primary': 'no',
                                                                                      'secondary': 'no',
                                                                                      'tertiary': {'poutcome': {'failure': 'no',
                                                                                                                'other': 'no',
                                                                                                                'success': {'age>38.0': {'no': 'no',
                                                                                                                                         'yes': 'yes'}},
                                                                                                                'unknown': {'marital': {'divorced': 'no',
                                                                                                                                        'married': {'campaign>2.0': {'no': {'age>38.0': {'no': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                  'yes': 'yes'}},
                                                                                                                                                                                         'yes': {'balance>452.5': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                   'yes': 'no'}}}},
                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                        'single': 'no'}}}},
                                                                                      'unknown': 'yes'}},
                                                         'retired': {'housing': {'no': 'no',
                                                                                 'yes': {'education': {'primary': 'yes',
                                                                                                       'secondary': 'no',
                                                                                                       'tertiary': 'yes',
                                                                                                       'unknown': 'yes'}}}},
                                                         'self-employed': 'no',
                                                         'services': 'no',
                                                         'student': {'housing': {'no': 'no',
                                                                                 'yes': 'yes'}},
                                                         'technician': 'no',
                                                         'unemployed': {'day>16.0': {'no': 'no',
                                                                                     'yes': 'yes'}},
                                                         'unknown': 'no'}},
                                         'jan': {'job': {'admin.': 'no',
                                                         'blue-collar': 'no',
                                                         'entrepreneur': 'no',
                                                         'housemaid': 'no',
                                                         'management': {'marital': {'divorced': 'no',
                                                                                    'married': 'no',
                                                                                    'single': {'age>38.0': {'no': 'no',
                                                                                                            'yes': 'yes'}}}},
                                                         'retired': 'no',
                                                         'self-employed': 'no',
                                                         'services': 'no',
                                                         'student': 'no',
                                                         'technician': 'no',
                                                         'unemployed': 'no',
                                                         'unknown': 'no'}},
                                         'jul': {'day>16.0': {'no': {'job': {'admin.': {'contact': {'cellular': 'no',
                                                                                                    'telephone': 'no',
                                                                                                    'unknown': {'age>38.0': {'no': 'no',
                                                                                                                             'yes': 'yes'}}}},
                                                                             'blue-collar': {'contact': {'cellular': 'no',
                                                                                                         'telephone': {'age>38.0': {'no': 'no',
                                                                                                                                    'yes': {'loan': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                             'yes': 'no'}},
                                                                                                                                                     'yes': 'no'}}}},
                                                                                                         'unknown': 'no'}},
                                                                             'entrepreneur': {'campaign>2.0': {'no': 'no',
                                                                                                               'yes': {'balance>452.5': {'no': 'no',
                                                                                                                                         'yes': 'yes'}}}},
                                                                             'housemaid': 'no',
                                                                             'management': {'housing': {'no': {'balance>452.5': {'no': 'no',
                                                                                                                                 'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                      'yes': {'loan': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                                                                          'married': {'education': {'primary': 'yes',
                                                                                                                                                                                                                    'secondary': 'yes',
                                                                                                                                                                                                                    'tertiary': {'default': {'no': {'contact': {'cellular': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': {'poutcome': {'failure': 'yes',
                                                                                                                                                                                                                                                                                                                                                                'other': 'yes',
                                                                                                                                                                                                                                                                                                                                                                'success': 'yes',
                                                                                                                                                                                                                                                                                                                                                                'unknown': 'yes'}},
                                                                                                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                                'telephone': 'yes',
                                                                                                                                                                                                                                                                'unknown': 'yes'}},
                                                                                                                                                                                                                                             'yes': 'yes'}},
                                                                                                                                                                                                                    'unknown': 'yes'}},
                                                                                                                                                                                          'single': 'yes'}},
                                                                                                                                                                       'yes': 'no'}}}}}},
                                                                                                        'yes': 'no'}},
                                                                             'retired': 'no',
                                                                             'self-employed': 'no',
                                                                             'services': 'no',
                                                                             'student': 'no',
                                                                             'technician': 'no',
                                                                             'unemployed': 'no',
                                                                             'unknown': 'no'}},
                                                              'yes': 'no'}},
                                         'jun': {'contact': {'cellular': {'job': {'admin.': {'loan': {'no': 'yes',
                                                                                                      'yes': 'no'}},
                                                                                  'blue-collar': {'age>38.0': {'no': 'yes',
                                                                                                               'yes': {'day>16.0': {'no': 'no',
                                                                                                                                    'yes': 'yes'}}}},
                                                                                  'entrepreneur': 'yes',
                                                                                  'housemaid': 'no',
                                                                                  'management': {'marital': {'divorced': 'no',
                                                                                                             'married': 'no',
                                                                                                             'single': {'pdays>-1.0': {'no': {'age>38.0': {'no': {'education': {'primary': 'yes',
                                                                                                                                                                                'secondary': 'yes',
                                                                                                                                                                                'tertiary': 'yes',
                                                                                                                                                                                'unknown': 'no'}},
                                                                                                                                                           'yes': 'no'}},
                                                                                                                                       'yes': 'no'}}}},
                                                                                  'retired': 'no',
                                                                                  'self-employed': 'no',
                                                                                  'services': 'yes',
                                                                                  'student': 'no',
                                                                                  'technician': {'balance>452.5': {'no': 'no',
                                                                                                                   'yes': {'age>38.0': {'no': 'no',
                                                                                                                                        'yes': 'yes'}}}},
                                                                                  'unemployed': 'no',
                                                                                  'unknown': 'no'}},
                                                             'telephone': 'no',
                                                             'unknown': 'no'}},
                                         'mar': {'job': {'admin.': 'no',
                                                         'blue-collar': 'no',
                                                         'entrepreneur': 'yes',
                                                         'housemaid': 'no',
                                                         'management': {'pdays>-1.0': {'no': {'day>16.0': {'no': {'age>38.0': {'no': 'no',
                                                                                                                               'yes': {'balance>452.5': {'no': 'no',
                                                                                                                                                         'yes': 'yes'}}}},
                                                                                                           'yes': 'yes'}},
                                                                                       'yes': 'no'}},
                                                         'retired': {'contact': {'cellular': 'yes',
                                                                                 'telephone': 'no',
                                                                                 'unknown': 'yes'}},
                                                         'self-employed': 'no',
                                                         'services': 'yes',
                                                         'student': {'balance>452.5': {'no': 'yes',
                                                                                       'yes': 'no'}},
                                                         'technician': 'yes',
                                                         'unemployed': {'marital': {'divorced': 'yes',
                                                                                    'married': 'yes',
                                                                                    'single': 'no'}},
                                                         'unknown': 'no'}},
                                         'may': {'job': {'admin.': {'education': {'primary': 'no',
                                                                                  'secondary': 'no',
                                                                                  'tertiary': {'balance>452.5': {'no': {'marital': {'divorced': 'no',
                                                                                                                                    'married': 'no',
                                                                                                                                    'single': 'yes'}},
                                                                                                                 'yes': 'no'}},
                                                                                  'unknown': 'no'}},
                                                         'blue-collar': 'no',
                                                         'entrepreneur': 'no',
                                                         'housemaid': 'no',
                                                         'management': {'poutcome': {'failure': 'no',
                                                                                     'other': 'no',
                                                                                     'success': {'housing': {'no': 'yes',
                                                                                                             'yes': 'no'}},
                                                                                     'unknown': 'no'}},
                                                         'retired': 'no',
                                                         'self-employed': 'no',
                                                         'services': {'marital': {'divorced': 'no',
                                                                                  'married': 'no',
                                                                                  'single': {'day>16.0': {'no': 'no',
                                                                                                          'yes': {'education': {'primary': 'no',
                                                                                                                                'secondary': {'contact': {'cellular': {'age>38.0': {'no': {'default': {'no': {'balance>452.5': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                   'yes': {'loan': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': {'poutcome': {'failure': 'yes',
                                                                                                                                                                                                                                                                                                                                                              'other': 'yes',
                                                                                                                                                                                                                                                                                                                                                              'success': 'yes',
                                                                                                                                                                                                                                                                                                                                                              'unknown': 'yes'}},
                                                                                                                                                                                                                                                                                                                                          'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                  'yes': 'yes'}},
                                                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                                                    'yes': 'yes'}}}},
                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                          'telephone': 'no',
                                                                                                                                                          'unknown': 'no'}},
                                                                                                                                'tertiary': 'no',
                                                                                                                                'unknown': 'yes'}}}}}},
                                                         'student': 'no',
                                                         'technician': 'no',
                                                         'unemployed': {'housing': {'no': {'contact': {'cellular': 'yes',
                                                                                                       'telephone': 'no',
                                                                                                       'unknown': 'no'}},
                                                                                    'yes': 'no'}},
                                                         'unknown': 'no'}},
                                         'nov': {'day>16.0': {'no': {'job': {'admin.': 'no',
                                                                             'blue-collar': {'age>38.0': {'no': 'yes',
                                                                                                          'yes': 'no'}},
                                                                             'entrepreneur': 'yes',
                                                                             'housemaid': 'no',
                                                                             'management': {'pdays>-1.0': {'no': 'no',
                                                                                                           'yes': 'yes'}},
                                                                             'retired': 'yes',
                                                                             'self-employed': 'no',
                                                                             'services': 'no',
                                                                             'student': 'no',
                                                                             'technician': 'no',
                                                                             'unemployed': 'no',
                                                                             'unknown': 'no'}},
                                                              'yes': {'contact': {'cellular': 'no',
                                                                                  'telephone': 'no',
                                                                                  'unknown': {'job': {'admin.': 'no',
                                                                                                      'blue-collar': 'yes',
                                                                                                      'entrepreneur': 'no',
                                                                                                      'housemaid': 'no',
                                                                                                      'management': 'no',
                                                                                                      'retired': 'no',
                                                                                                      'self-employed': 'no',
                                                                                                      'services': 'no',
                                                                                                      'student': 'no',
                                                                                                      'technician': 'no',
                                                                                                      'unemployed': 'no',
                                                                                                      'unknown': 'no'}}}}}},
                                         'oct': {'education': {'primary': 'no',
                                                               'secondary': {'balance>452.5': {'no': {'job': {'admin.': 'no',
                                                                                                              'blue-collar': 'yes',
                                                                                                              'entrepreneur': 'yes',
                                                                                                              'housemaid': 'yes',
                                                                                                              'management': 'yes',
                                                                                                              'retired': 'yes',
                                                                                                              'self-employed': 'yes',
                                                                                                              'services': 'yes',
                                                                                                              'student': 'yes',
                                                                                                              'technician': 'yes',
                                                                                                              'unemployed': 'yes',
                                                                                                              'unknown': 'yes'}},
                                                                                               'yes': {'housing': {'no': 'no',
                                                                                                                   'yes': 'yes'}}}},
                                                               'tertiary': {'job': {'admin.': 'no',
                                                                                    'blue-collar': 'no',
                                                                                    'entrepreneur': 'no',
                                                                                    'housemaid': 'yes',
                                                                                    'management': {'age>38.0': {'no': 'no',
                                                                                                                'yes': {'campaign>2.0': {'no': 'yes',
                                                                                                                                         'yes': 'no'}}}},
                                                                                    'retired': 'no',
                                                                                    'self-employed': 'no',
                                                                                    'services': 'no',
                                                                                    'student': 'no',
                                                                                    'technician': 'yes',
                                                                                    'unemployed': 'no',
                                                                                    'unknown': 'no'}},
                                                               'unknown': 'no'}},
                                         'sep': {'education': {'primary': 'no',
                                                               'secondary': {'job': {'admin.': 'no',
                                                                                     'blue-collar': 'no',
                                                                                     'entrepreneur': 'no',
                                                                                     'housemaid': 'no',
                                                                                     'management': 'no',
                                                                                     'retired': {'balance>452.5': {'no': 'no',
                                                                                                                   'yes': 'yes'}},
                                                                                     'self-employed': 'no',
                                                                                     'services': 'no',
                                                                                     'student': 'no',
                                                                                     'technician': {'day>16.0': {'no': 'no',
                                                                                                                 'yes': 'yes'}},
                                                                                     'unemployed': 'no',
                                                                                     'unknown': 'no'}},
                                                               'tertiary': {'balance>452.5': {'no': 'yes',
                                                                                              'yes': {'campaign>2.0': {'no': 'no',
                                                                                                                       'yes': 'yes'}}}},
                                                               'unknown': 'yes'}}}},
                        'yes': {'poutcome': {'failure': {'month': {'apr': {'job': {'admin.': {'age>38.0': {'no': 'no',
                                                                                                           'yes': 'yes'}},
                                                                                   'blue-collar': {'marital': {'divorced': 'no',
                                                                                                               'married': 'no',
                                                                                                               'single': {'loan': {'no': {'age>38.0': {'no': {'education': {'primary': 'yes',
                                                                                                                                                                            'secondary': {'default': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                               'yes': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                   'yes': {'contact': {'cellular': {'day>16.0': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                               'yes': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                 'yes': 'yes'}},
                                                                                                                                                                                                                                                                       'telephone': 'yes',
                                                                                                                                                                                                                                                                       'unknown': 'yes'}}}}}},
                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                            'tertiary': 'yes',
                                                                                                                                                                            'unknown': 'yes'}},
                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                   'yes': 'no'}}}},
                                                                                   'entrepreneur': 'no',
                                                                                   'housemaid': 'no',
                                                                                   'management': {'age>38.0': {'no': {'day>16.0': {'no': 'yes',
                                                                                                                                   'yes': {'education': {'primary': 'no',
                                                                                                                                                         'secondary': 'yes',
                                                                                                                                                         'tertiary': 'no',
                                                                                                                                                         'unknown': 'no'}}}},
                                                                                                               'yes': 'no'}},
                                                                                   'retired': 'no',
                                                                                   'self-employed': {'marital': {'divorced': 'yes',
                                                                                                                 'married': 'no',
                                                                                                                 'single': 'yes'}},
                                                                                   'services': 'no',
                                                                                   'student': 'no',
                                                                                   'technician': {'campaign>2.0': {'no': 'no',
                                                                                                                   'yes': 'yes'}},
                                                                                   'unemployed': 'no',
                                                                                   'unknown': 'no'}},
                                                                   'aug': {'job': {'admin.': 'yes',
                                                                                   'blue-collar': {'marital': {'divorced': 'yes',
                                                                                                               'married': 'yes',
                                                                                                               'single': 'no'}},
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': {'education': {'primary': 'no',
                                                                                                               'secondary': 'yes',
                                                                                                               'tertiary': 'no',
                                                                                                               'unknown': 'no'}},
                                                                                   'management': 'yes',
                                                                                   'retired': 'no',
                                                                                   'self-employed': 'yes',
                                                                                   'services': 'yes',
                                                                                   'student': 'yes',
                                                                                   'technician': {'age>38.0': {'no': 'yes',
                                                                                                               'yes': 'no'}},
                                                                                   'unemployed': 'no',
                                                                                   'unknown': 'yes'}},
                                                                   'dec': {'job': {'admin.': 'yes',
                                                                                   'blue-collar': 'no',
                                                                                   'entrepreneur': 'no',
                                                                                   'housemaid': 'no',
                                                                                   'management': 'no',
                                                                                   'retired': 'no',
                                                                                   'self-employed': 'no',
                                                                                   'services': 'no',
                                                                                   'student': 'no',
                                                                                   'technician': 'no',
                                                                                   'unemployed': 'no',
                                                                                   'unknown': 'no'}},
                                                                   'feb': {'job': {'admin.': 'no',
                                                                                   'blue-collar': 'no',
                                                                                   'entrepreneur': 'no',
                                                                                   'housemaid': 'no',
                                                                                   'management': 'no',
                                                                                   'retired': 'no',
                                                                                   'self-employed': 'no',
                                                                                   'services': 'yes',
                                                                                   'student': 'no',
                                                                                   'technician': 'no',
                                                                                   'unemployed': 'no',
                                                                                   'unknown': 'no'}},
                                                                   'jan': 'no',
                                                                   'jul': {'job': {'admin.': 'yes',
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'yes',
                                                                                   'management': {'age>38.0': {'no': 'yes',
                                                                                                               'yes': 'no'}},
                                                                                   'retired': 'no',
                                                                                   'self-employed': 'yes',
                                                                                   'services': 'yes',
                                                                                   'student': 'no',
                                                                                   'technician': 'yes',
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'yes'}},
                                                                   'jun': {'job': {'admin.': {'age>38.0': {'no': 'yes',
                                                                                                           'yes': 'no'}},
                                                                                   'blue-collar': 'no',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'yes',
                                                                                   'management': {'housing': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                      'yes': 'yes'}},
                                                                                                              'yes': 'yes'}},
                                                                                   'retired': 'yes',
                                                                                   'self-employed': 'yes',
                                                                                   'services': 'no',
                                                                                   'student': 'yes',
                                                                                   'technician': {'age>38.0': {'no': 'no',
                                                                                                               'yes': 'yes'}},
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'yes'}},
                                                                   'mar': {'education': {'primary': 'no',
                                                                                         'secondary': 'yes',
                                                                                         'tertiary': 'yes',
                                                                                         'unknown': 'yes'}},
                                                                   'may': {'job': {'admin.': {'day>16.0': {'no': {'marital': {'divorced': 'no',
                                                                                                                              'married': 'no',
                                                                                                                              'single': {'campaign>2.0': {'no': 'no',
                                                                                                                                                          'yes': 'yes'}}}},
                                                                                                           'yes': 'yes'}},
                                                                                   'blue-collar': {'housing': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                  'married': 'no',
                                                                                                                                  'single': 'yes'}},
                                                                                                               'yes': 'no'}},
                                                                                   'entrepreneur': {'marital': {'divorced': 'no',
                                                                                                                'married': 'no',
                                                                                                                'single': 'yes'}},
                                                                                   'housemaid': 'no',
                                                                                   'management': {'balance>452.5': {'no': {'age>38.0': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                                           'married': 'yes',
                                                                                                                                                           'single': 'no'}},
                                                                                                                                        'yes': 'no'}},
                                                                                                                    'yes': 'no'}},
                                                                                   'retired': 'no',
                                                                                   'self-employed': 'no',
                                                                                   'services': {'day>16.0': {'no': {'age>38.0': {'no': {'balance>452.5': {'no': 'no',
                                                                                                                                                          'yes': {'marital': {'divorced': 'yes',
                                                                                                                                                                              'married': {'education': {'primary': 'yes',
                                                                                                                                                                                                        'secondary': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                     'yes': {'loan': {'no': {'contact': {'cellular': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                             'yes': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                         'telephone': 'yes',
                                                                                                                                                                                                                                                                                         'unknown': 'yes'}},
                                                                                                                                                                                                                                                                      'yes': 'yes'}}}},
                                                                                                                                                                                                                                  'yes': 'yes'}},
                                                                                                                                                                                                        'tertiary': 'yes',
                                                                                                                                                                                                        'unknown': 'yes'}},
                                                                                                                                                                              'single': 'yes'}}}},
                                                                                                                                 'yes': 'no'}},
                                                                                                             'yes': 'yes'}},
                                                                                   'student': 'no',
                                                                                   'technician': {'loan': {'no': 'no',
                                                                                                           'yes': 'yes'}},
                                                                                   'unemployed': 'no',
                                                                                   'unknown': 'no'}},
                                                                   'nov': {'job': {'admin.': {'marital': {'divorced': 'no',
                                                                                                          'married': 'no',
                                                                                                          'single': 'yes'}},
                                                                                   'blue-collar': 'no',
                                                                                   'entrepreneur': 'no',
                                                                                   'housemaid': 'no',
                                                                                   'management': {'day>16.0': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                  'married': {'age>38.0': {'no': 'no',
                                                                                                                                                           'yes': {'education': {'primary': 'yes',
                                                                                                                                                                                 'secondary': 'no',
                                                                                                                                                                                 'tertiary': 'yes',
                                                                                                                                                                                 'unknown': 'yes'}}}},
                                                                                                                                  'single': 'yes'}},
                                                                                                               'yes': 'no'}},
                                                                                   'retired': 'no',
                                                                                   'self-employed': 'no',
                                                                                   'services': 'no',
                                                                                   'student': 'no',
                                                                                   'technician': {'education': {'primary': 'no',
                                                                                                                'secondary': 'no',
                                                                                                                'tertiary': 'yes',
                                                                                                                'unknown': 'no'}},
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'no'}},
                                                                   'oct': {'job': {'admin.': 'no',
                                                                                   'blue-collar': 'no',
                                                                                   'entrepreneur': 'no',
                                                                                   'housemaid': 'no',
                                                                                   'management': 'yes',
                                                                                   'retired': 'yes',
                                                                                   'self-employed': 'no',
                                                                                   'services': 'no',
                                                                                   'student': 'no',
                                                                                   'technician': {'housing': {'no': 'yes',
                                                                                                              'yes': 'no'}},
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'no'}},
                                                                   'sep': {'job': {'admin.': 'no',
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'no',
                                                                                   'housemaid': 'no',
                                                                                   'management': {'marital': {'divorced': 'yes',
                                                                                                              'married': 'no',
                                                                                                              'single': {'age>38.0': {'no': 'no',
                                                                                                                                      'yes': 'yes'}}}},
                                                                                   'retired': 'no',
                                                                                   'self-employed': 'no',
                                                                                   'services': 'no',
                                                                                   'student': 'no',
                                                                                   'technician': 'no',
                                                                                   'unemployed': {'age>38.0': {'no': 'yes',
                                                                                                               'yes': 'no'}},
                                                                                   'unknown': 'no'}}}},
                                             'other': {'job': {'admin.': {'month': {'apr': 'no',
                                                                                    'aug': 'no',
                                                                                    'dec': 'yes',
                                                                                    'feb': 'no',
                                                                                    'jan': 'no',
                                                                                    'jul': 'no',
                                                                                    'jun': 'no',
                                                                                    'mar': 'no',
                                                                                    'may': 'no',
                                                                                    'nov': 'no',
                                                                                    'oct': 'no',
                                                                                    'sep': 'no'}},
                                                               'blue-collar': {'month': {'apr': {'balance>452.5': {'no': 'yes',
                                                                                                                   'yes': 'no'}},
                                                                                         'aug': 'no',
                                                                                         'dec': 'no',
                                                                                         'feb': 'no',
                                                                                         'jan': 'no',
                                                                                         'jul': 'no',
                                                                                         'jun': 'yes',
                                                                                         'mar': 'no',
                                                                                         'may': 'no',
                                                                                         'nov': 'no',
                                                                                         'oct': 'yes',
                                                                                         'sep': 'no'}},
                                                               'entrepreneur': 'no',
                                                               'housemaid': 'no',
                                                               'management': {'month': {'apr': {'age>38.0': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                      'yes': 'no'}},
                                                                                                             'yes': 'no'}},
                                                                                        'aug': {'age>38.0': {'no': 'yes',
                                                                                                             'yes': 'no'}},
                                                                                        'dec': 'yes',
                                                                                        'feb': {'marital': {'divorced': 'no',
                                                                                                            'married': 'yes',
                                                                                                            'single': 'no'}},
                                                                                        'jan': 'no',
                                                                                        'jul': 'yes',
                                                                                        'jun': 'no',
                                                                                        'mar': 'no',
                                                                                        'may': {'education': {'primary': 'yes',
                                                                                                              'secondary': 'yes',
                                                                                                              'tertiary': {'balance>452.5': {'no': 'no',
                                                                                                                                             'yes': {'marital': {'divorced': 'yes',
                                                                                                                                                                 'married': {'age>38.0': {'no': 'no',
                                                                                                                                                                                          'yes': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                 'yes': {'loan': {'no': {'contact': {'cellular': {'day>16.0': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                             'yes': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                                     'telephone': 'yes',
                                                                                                                                                                                                                                                                     'unknown': 'yes'}},
                                                                                                                                                                                                                                                  'yes': 'yes'}}}},
                                                                                                                                                                                                              'yes': 'yes'}}}},
                                                                                                                                                                 'single': 'yes'}}}},
                                                                                                              'unknown': 'yes'}},
                                                                                        'nov': 'no',
                                                                                        'oct': 'yes',
                                                                                        'sep': 'no'}},
                                                               'retired': {'month': {'apr': 'no',
                                                                                     'aug': 'no',
                                                                                     'dec': 'yes',
                                                                                     'feb': 'no',
                                                                                     'jan': 'no',
                                                                                     'jul': 'no',
                                                                                     'jun': 'no',
                                                                                     'mar': 'no',
                                                                                     'may': 'yes',
                                                                                     'nov': 'no',
                                                                                     'oct': 'no',
                                                                                     'sep': 'no'}},
                                                               'self-employed': {'marital': {'divorced': 'no',
                                                                                             'married': 'no',
                                                                                             'single': 'yes'}},
                                                               'services': {'age>38.0': {'no': 'no',
                                                                                         'yes': {'marital': {'divorced': 'no',
                                                                                                             'married': 'no',
                                                                                                             'single': {'education': {'primary': 'yes',
                                                                                                                                      'secondary': {'default': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                         'yes': {'housing': {'no': 'yes',
                                                                                                                                                                                                             'yes': {'loan': {'no': {'contact': {'cellular': {'day>16.0': {'no': {'month': {'apr': 'yes',
                                                                                                                                                                                                                                                                                            'aug': 'yes',
                                                                                                                                                                                                                                                                                            'dec': 'yes',
                                                                                                                                                                                                                                                                                            'feb': 'yes',
                                                                                                                                                                                                                                                                                            'jan': 'yes',
                                                                                                                                                                                                                                                                                            'jul': 'yes',
                                                                                                                                                                                                                                                                                            'jun': 'yes',
                                                                                                                                                                                                                                                                                            'mar': 'yes',
                                                                                                                                                                                                                                                                                            'may': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                     'yes': {'pdays>-1.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                            'yes': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                     'yes': 'yes'}}}}}},
                                                                                                                                                                                                                                                                                            'nov': 'yes',
                                                                                                                                                                                                                                                                                            'oct': 'yes',
                                                                                                                                                                                                                                                                                            'sep': 'yes'}},
                                                                                                                                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                                                                                                                                 'telephone': 'yes',
                                                                                                                                                                                                                                                 'unknown': 'yes'}},
                                                                                                                                                                                                                              'yes': 'yes'}}}}}},
                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                      'tertiary': 'yes',
                                                                                                                                      'unknown': 'yes'}}}}}},
                                                               'student': 'no',
                                                               'technician': {'month': {'apr': 'yes',
                                                                                        'aug': 'yes',
                                                                                        'dec': 'yes',
                                                                                        'feb': 'no',
                                                                                        'jan': 'no',
                                                                                        'jul': 'yes',
                                                                                        'jun': 'yes',
                                                                                        'mar': 'no',
                                                                                        'may': {'housing': {'no': 'yes',
                                                                                                            'yes': 'no'}},
                                                                                        'nov': {'marital': {'divorced': 'yes',
                                                                                                            'married': 'yes',
                                                                                                            'single': 'no'}},
                                                                                        'oct': 'yes',
                                                                                        'sep': 'yes'}},
                                                               'unemployed': 'no',
                                                               'unknown': 'no'}},
                                             'success': {'month': {'apr': {'education': {'primary': 'yes',
                                                                                         'secondary': 'yes',
                                                                                         'tertiary': {'housing': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                     'married': {'job': {'admin.': 'no',
                                                                                                                                                         'blue-collar': 'no',
                                                                                                                                                         'entrepreneur': 'no',
                                                                                                                                                         'housemaid': 'no',
                                                                                                                                                         'management': {'age>38.0': {'no': {'default': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                                 'yes': {'loan': {'no': {'contact': {'cellular': {'day>16.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                               'yes': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                              'yes': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                     'telephone': 'yes',
                                                                                                                                                                                                                                                                     'unknown': 'yes'}},
                                                                                                                                                                                                                                                  'yes': 'yes'}}}},
                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                         'retired': 'no',
                                                                                                                                                         'self-employed': 'no',
                                                                                                                                                         'services': 'no',
                                                                                                                                                         'student': 'no',
                                                                                                                                                         'technician': 'no',
                                                                                                                                                         'unemployed': 'no',
                                                                                                                                                         'unknown': 'no'}},
                                                                                                                                     'single': 'yes'}},
                                                                                                                  'yes': 'no'}},
                                                                                         'unknown': 'yes'}},
                                                                   'aug': {'education': {'primary': 'no',
                                                                                         'secondary': 'yes',
                                                                                         'tertiary': 'no',
                                                                                         'unknown': 'yes'}},
                                                                   'dec': 'yes',
                                                                   'feb': {'age>38.0': {'no': 'yes',
                                                                                        'yes': {'education': {'primary': 'yes',
                                                                                                              'secondary': 'no',
                                                                                                              'tertiary': 'no',
                                                                                                              'unknown': 'yes'}}}},
                                                                   'jan': {'job': {'admin.': 'yes',
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'yes',
                                                                                   'management': 'yes',
                                                                                   'retired': 'yes',
                                                                                   'self-employed': 'no',
                                                                                   'services': 'yes',
                                                                                   'student': 'yes',
                                                                                   'technician': 'yes',
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'yes'}},
                                                                   'jul': {'age>38.0': {'no': 'no',
                                                                                        'yes': 'yes'}},
                                                                   'jun': {'job': {'admin.': 'yes',
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'yes',
                                                                                   'management': 'yes',
                                                                                   'retired': 'no',
                                                                                   'self-employed': 'yes',
                                                                                   'services': 'yes',
                                                                                   'student': 'yes',
                                                                                   'technician': 'yes',
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'yes'}},
                                                                   'mar': 'yes',
                                                                   'may': {'contact': {'cellular': {'job': {'admin.': 'yes',
                                                                                                            'blue-collar': {'age>38.0': {'no': 'no',
                                                                                                                                         'yes': 'yes'}},
                                                                                                            'entrepreneur': 'yes',
                                                                                                            'housemaid': 'yes',
                                                                                                            'management': 'yes',
                                                                                                            'retired': 'yes',
                                                                                                            'self-employed': 'yes',
                                                                                                            'services': 'yes',
                                                                                                            'student': 'yes',
                                                                                                            'technician': {'balance>452.5': {'no': 'yes',
                                                                                                                                             'yes': 'no'}},
                                                                                                            'unemployed': 'yes',
                                                                                                            'unknown': 'yes'}},
                                                                                       'telephone': 'no',
                                                                                       'unknown': 'yes'}},
                                                                   'nov': {'job': {'admin.': 'yes',
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'yes',
                                                                                   'management': {'marital': {'divorced': 'yes',
                                                                                                              'married': 'yes',
                                                                                                              'single': 'no'}},
                                                                                   'retired': 'yes',
                                                                                   'self-employed': 'yes',
                                                                                   'services': 'yes',
                                                                                   'student': 'yes',
                                                                                   'technician': 'no',
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'yes'}},
                                                                   'oct': {'job': {'admin.': 'yes',
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'yes',
                                                                                   'management': 'yes',
                                                                                   'retired': 'yes',
                                                                                   'self-employed': 'yes',
                                                                                   'services': 'yes',
                                                                                   'student': 'no',
                                                                                   'technician': {'age>38.0': {'no': 'yes',
                                                                                                               'yes': 'no'}},
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'yes'}},
                                                                   'sep': 'yes'}},
                                             'unknown': {'month': {'apr': {'housing': {'no': {'job': {'admin.': {'marital': {'divorced': 'yes',
                                                                                                                             'married': {'age>38.0': {'no': {'loan': {'no': 'yes',
                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                      'yes': 'no'}},
                                                                                                                             'single': 'yes'}},
                                                                                                      'blue-collar': {'education': {'primary': {'age>38.0': {'no': 'no',
                                                                                                                                                             'yes': 'yes'}},
                                                                                                                                    'secondary': 'no',
                                                                                                                                    'tertiary': 'yes',
                                                                                                                                    'unknown': 'no'}},
                                                                                                      'entrepreneur': 'yes',
                                                                                                      'housemaid': 'yes',
                                                                                                      'management': {'marital': {'divorced': 'yes',
                                                                                                                                 'married': {'age>38.0': {'no': 'no',
                                                                                                                                                          'yes': {'education': {'primary': 'yes',
                                                                                                                                                                                'secondary': 'no',
                                                                                                                                                                                'tertiary': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                               'yes': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                        'yes': 'yes'}}}},
                                                                                                                                                                                'unknown': 'yes'}}}},
                                                                                                                                 'single': 'yes'}},
                                                                                                      'retired': {'marital': {'divorced': {'education': {'primary': 'yes',
                                                                                                                                                         'secondary': 'yes',
                                                                                                                                                         'tertiary': 'no',
                                                                                                                                                         'unknown': 'yes'}},
                                                                                                                              'married': {'education': {'primary': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                 'yes': {'default': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                              'yes': {'loan': {'no': {'contact': {'cellular': 'yes',
                                                                                                                                                                                                                                                                  'telephone': {'day>16.0': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                                                                                                                             'yes': 'yes'}},
                                                                                                                                                                                                                                                                  'unknown': 'yes'}},
                                                                                                                                                                                                                                               'yes': 'yes'}}}},
                                                                                                                                                                                                     'yes': 'yes'}}}},
                                                                                                                                                        'secondary': 'yes',
                                                                                                                                                        'tertiary': 'yes',
                                                                                                                                                        'unknown': 'yes'}},
                                                                                                                              'single': 'no'}},
                                                                                                      'self-employed': 'no',
                                                                                                      'services': 'no',
                                                                                                      'student': {'education': {'primary': 'no',
                                                                                                                                'secondary': 'yes',
                                                                                                                                'tertiary': 'yes',
                                                                                                                                'unknown': 'yes'}},
                                                                                                      'technician': {'age>38.0': {'no': 'yes',
                                                                                                                                  'yes': 'no'}},
                                                                                                      'unemployed': 'yes',
                                                                                                      'unknown': 'yes'}},
                                                                                       'yes': {'job': {'admin.': 'no',
                                                                                                       'blue-collar': {'education': {'primary': 'no',
                                                                                                                                     'secondary': {'loan': {'no': {'campaign>2.0': {'no': {'balance>452.5': {'no': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                                                                             'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                                                  'yes': {'day>16.0': {'no': {'marital': {'divorced': 'no',
                                                                                                                                                                                                                                                                          'married': {'default': {'no': {'contact': {'cellular': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                         'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                     'telephone': 'no',
                                                                                                                                                                                                                                                                                                                     'unknown': 'no'}},
                                                                                                                                                                                                                                                                                                  'yes': 'no'}},
                                                                                                                                                                                                                                                                          'single': 'no'}},
                                                                                                                                                                                                                                                       'yes': 'yes'}}}}}},
                                                                                                                                                                                    'yes': 'no'}},
                                                                                                                                                            'yes': 'no'}},
                                                                                                                                     'tertiary': 'no',
                                                                                                                                     'unknown': 'no'}},
                                                                                                       'entrepreneur': {'loan': {'no': {'default': {'no': 'yes',
                                                                                                                                                    'yes': 'no'}},
                                                                                                                                 'yes': 'no'}},
                                                                                                       'housemaid': 'no',
                                                                                                       'management': {'education': {'primary': 'no',
                                                                                                                                    'secondary': 'no',
                                                                                                                                    'tertiary': {'loan': {'no': {'age>38.0': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                      'yes': {'marital': {'divorced': 'yes',
                                                                                                                                                                                                                          'married': 'no',
                                                                                                                                                                                                                          'single': 'yes'}}}},
                                                                                                                                                                              'yes': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                        'yes': 'no'}}}},
                                                                                                                                                          'yes': 'no'}},
                                                                                                                                    'unknown': 'no'}},
                                                                                                       'retired': 'no',
                                                                                                       'self-employed': 'no',
                                                                                                       'services': {'marital': {'divorced': 'no',
                                                                                                                                'married': 'no',
                                                                                                                                'single': {'age>38.0': {'no': 'no',
                                                                                                                                                        'yes': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                  'yes': 'no'}}}}}},
                                                                                                       'student': 'no',
                                                                                                       'technician': {'balance>452.5': {'no': 'no',
                                                                                                                                        'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                             'yes': {'education': {'primary': 'no',
                                                                                                                                                                                   'secondary': 'yes',
                                                                                                                                                                                   'tertiary': 'yes',
                                                                                                                                                                                   'unknown': 'yes'}}}}}},
                                                                                                       'unemployed': 'no',
                                                                                                       'unknown': 'no'}}}},
                                                                   'aug': {'education': {'primary': 'no',
                                                                                         'secondary': {'job': {'admin.': {'day>16.0': {'no': 'no',
                                                                                                                                       'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                            'yes': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                     'yes': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                               'yes': {'marital': {'divorced': 'yes',
                                                                                                                                                                                                                                   'married': {'default': {'no': {'housing': {'no': {'loan': {'no': {'contact': {'cellular': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                             'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                 'telephone': 'yes',
                                                                                                                                                                                                                                                                                                                 'unknown': 'yes'}},
                                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                                                                                                                   'single': 'yes'}}}}}}}}}},
                                                                                                               'blue-collar': {'balance>452.5': {'no': 'no',
                                                                                                                                                 'yes': {'contact': {'cellular': {'campaign>2.0': {'no': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                                                       'yes': {'marital': {'divorced': 'yes',
                                                                                                                                                                                                                                           'married': {'default': {'no': {'housing': {'no': {'loan': {'no': {'day>16.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                          'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                                                                           'single': 'yes'}}}},
                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                     'telephone': 'no',
                                                                                                                                                                     'unknown': 'yes'}}}},
                                                                                                               'entrepreneur': 'no',
                                                                                                               'housemaid': 'no',
                                                                                                               'management': 'no',
                                                                                                               'retired': {'day>16.0': {'no': 'no',
                                                                                                                                        'yes': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                 'yes': 'no'}}}},
                                                                                                               'self-employed': 'yes',
                                                                                                               'services': 'no',
                                                                                                               'student': 'no',
                                                                                                               'technician': {'loan': {'no': {'day>16.0': {'no': {'default': {'no': {'marital': {'divorced': {'housing': {'no': 'no',
                                                                                                                                                                                                                          'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                                                               'yes': 'yes'}}}},
                                                                                                                                                                                                 'married': {'housing': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                 'yes': {'age>38.0': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                                                                                               'yes': {'contact': {'cellular': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                   'telephone': 'yes',
                                                                                                                                                                                                                                                                                                                   'unknown': 'yes'}}}},
                                                                                                                                                                                                                                                                      'yes': 'no'}}}},
                                                                                                                                                                                                                         'yes': 'no'}},
                                                                                                                                                                                                 'single': {'housing': {'no': {'campaign>2.0': {'no': {'age>38.0': {'no': {'balance>452.5': {'no': {'contact': {'cellular': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                'telephone': 'yes',
                                                                                                                                                                                                                                                                                                                'unknown': 'yes'}},
                                                                                                                                                                                                                                                                                             'yes': 'no'}},
                                                                                                                                                                                                                                                                    'yes': 'no'}},
                                                                                                                                                                                                                                                'yes': 'no'}},
                                                                                                                                                                                                                        'yes': 'yes'}}}},
                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                           'yes': {'marital': {'divorced': 'no',
                                                                                                                                                                               'married': 'no',
                                                                                                                                                                               'single': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                           'yes': {'housing': {'no': {'age>38.0': {'no': {'default': {'no': {'balance>452.5': {'no': {'contact': {'cellular': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                  'telephone': 'no',
                                                                                                                                                                                                                                                                                                                  'unknown': 'no'}},
                                                                                                                                                                                                                                                                                               'yes': 'no'}},
                                                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                                                   'yes': 'no'}},
                                                                                                                                                                                                                               'yes': 'no'}}}}}}}},
                                                                                                                                       'yes': 'no'}},
                                                                                                               'unemployed': 'no',
                                                                                                               'unknown': 'no'}},
                                                                                         'tertiary': {'contact': {'cellular': {'marital': {'divorced': {'day>16.0': {'no': {'job': {'admin.': 'yes',
                                                                                                                                                                                    'blue-collar': 'yes',
                                                                                                                                                                                    'entrepreneur': 'yes',
                                                                                                                                                                                    'housemaid': 'yes',
                                                                                                                                                                                    'management': {'housing': {'no': {'age>38.0': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                                                   'yes': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                                            'yes': 'yes'}}}},
                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                    'retired': 'yes',
                                                                                                                                                                                    'self-employed': 'yes',
                                                                                                                                                                                    'services': 'yes',
                                                                                                                                                                                    'student': 'yes',
                                                                                                                                                                                    'technician': 'yes',
                                                                                                                                                                                    'unemployed': 'yes',
                                                                                                                                                                                    'unknown': 'yes'}},
                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                           'married': {'job': {'admin.': 'no',
                                                                                                                                                               'blue-collar': 'no',
                                                                                                                                                               'entrepreneur': {'age>38.0': {'no': 'no',
                                                                                                                                                                                             'yes': 'yes'}},
                                                                                                                                                               'housemaid': 'no',
                                                                                                                                                               'management': {'loan': {'no': {'age>38.0': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                               'yes': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                                                         'yes': {'housing': {'no': {'default': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                        'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'no'}}}},
                                                                                                                                                                                                                                                                                                'yes': 'no'}},
                                                                                                                                                                                                                                                                             'yes': 'no'}}}}}},
                                                                                                                                                                                                           'yes': {'day>16.0': {'no': {'balance>452.5': {'no': {'housing': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                    'yes': {'default': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'no'}}}},
                                                                                                                                                                                                                                                                            'yes': {'default': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                                                                                                                                                                'yes': 'no'}}}},
                                                                                                                                                                                                                                                         'yes': {'housing': {'no': {'campaign>2.0': {'no': {'default': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                                                                                                                                                                     'yes': {'default': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                         'yes': 'no'}}}},
                                                                                                                                                                                                                                                                             'yes': 'no'}}}},
                                                                                                                                                                                                                                'yes': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                                                          'yes': {'housing': {'no': {'default': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                         'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'no'}}}},
                                                                                                                                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                                                                                                                                              'yes': 'no'}}}}}}}},
                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                               'retired': 'no',
                                                                                                                                                               'self-employed': 'no',
                                                                                                                                                               'services': 'no',
                                                                                                                                                               'student': 'no',
                                                                                                                                                               'technician': {'age>38.0': {'no': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                    'yes': {'housing': {'no': {'day>16.0': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                    'yes': 'no'}},
                                                                                                                                                                                                                                                            'yes': 'no'}},
                                                                                                                                                                                                                                        'yes': 'no'}}}},
                                                                                                                                                                                           'yes': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                    'yes': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                         'yes': {'default': {'no': {'balance>452.5': {'no': {'housing': {'no': {'loan': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                             'yes': 'yes'}}}}}}}},
                                                                                                                                                               'unemployed': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                               'unknown': 'no'}},
                                                                                                                                           'single': {'job': {'admin.': 'no',
                                                                                                                                                              'blue-collar': 'no',
                                                                                                                                                              'entrepreneur': 'no',
                                                                                                                                                              'housemaid': 'no',
                                                                                                                                                              'management': {'balance>452.5': {'no': {'day>16.0': {'no': {'loan': {'no': {'campaign>2.0': {'no': {'age>38.0': {'no': {'default': {'no': {'housing': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                  'yes': 'yes'}},
                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                                                   'yes': 'no'}},
                                                                                                                                                                                                                   'yes': 'no'}},
                                                                                                                                                                                               'yes': {'housing': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                           'yes': {'age>38.0': {'no': {'day>16.0': {'no': {'default': {'no': {'loan': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                             'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                                                                                                                                                    'yes': {'default': {'no': {'loan': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                        'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                'yes': {'day>16.0': {'no': {'default': {'no': {'loan': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                     'yes': 'no'}}}}}},
                                                                                                                                                                                                                   'yes': 'yes'}}}},
                                                                                                                                                              'retired': 'yes',
                                                                                                                                                              'self-employed': 'yes',
                                                                                                                                                              'services': 'no',
                                                                                                                                                              'student': 'no',
                                                                                                                                                              'technician': {'age>38.0': {'no': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                   'yes': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                        'yes': {'campaign>2.0': {'no': {'default': {'no': {'housing': {'no': {'loan': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                             'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                                                                                                                                 'yes': 'no'}}}}}},
                                                                                                                                                                                          'yes': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                   'yes': 'yes'}}}},
                                                                                                                                                              'unemployed': 'no',
                                                                                                                                                              'unknown': 'no'}}}},
                                                                                                                  'telephone': 'no',
                                                                                                                  'unknown': 'yes'}},
                                                                                         'unknown': {'day>16.0': {'no': 'no',
                                                                                                                  'yes': {'job': {'admin.': 'no',
                                                                                                                                  'blue-collar': 'yes',
                                                                                                                                  'entrepreneur': 'yes',
                                                                                                                                  'housemaid': 'yes',
                                                                                                                                  'management': 'yes',
                                                                                                                                  'retired': 'yes',
                                                                                                                                  'self-employed': 'yes',
                                                                                                                                  'services': 'yes',
                                                                                                                                  'student': 'yes',
                                                                                                                                  'technician': 'yes',
                                                                                                                                  'unemployed': 'yes',
                                                                                                                                  'unknown': 'yes'}}}}}},
                                                                   'dec': {'job': {'admin.': 'yes',
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'no',
                                                                                   'management': 'yes',
                                                                                   'retired': 'yes',
                                                                                   'self-employed': {'education': {'primary': 'yes',
                                                                                                                   'secondary': 'no',
                                                                                                                   'tertiary': 'yes',
                                                                                                                   'unknown': 'yes'}},
                                                                                   'services': 'no',
                                                                                   'student': 'yes',
                                                                                   'technician': 'no',
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'yes'}},
                                                                   'feb': {'day>16.0': {'no': {'job': {'admin.': {'balance>452.5': {'no': {'contact': {'cellular': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                                       'telephone': 'no',
                                                                                                                                                       'unknown': 'yes'}},
                                                                                                                                    'yes': 'no'}},
                                                                                                       'blue-collar': {'education': {'primary': {'age>38.0': {'no': 'no',
                                                                                                                                                              'yes': {'marital': {'divorced': 'no',
                                                                                                                                                                                  'married': 'yes',
                                                                                                                                                                                  'single': 'yes'}}}},
                                                                                                                                     'secondary': 'no',
                                                                                                                                     'tertiary': 'no',
                                                                                                                                     'unknown': 'no'}},
                                                                                                       'entrepreneur': {'education': {'primary': 'no',
                                                                                                                                      'secondary': 'yes',
                                                                                                                                      'tertiary': 'no',
                                                                                                                                      'unknown': 'no'}},
                                                                                                       'housemaid': {'marital': {'divorced': 'yes',
                                                                                                                                 'married': 'yes',
                                                                                                                                 'single': 'no'}},
                                                                                                       'management': {'loan': {'no': 'no',
                                                                                                                               'yes': {'age>38.0': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                                                       'married': 'yes',
                                                                                                                                                                       'single': 'no'}},
                                                                                                                                                    'yes': 'no'}}}},
                                                                                                       'retired': {'contact': {'cellular': {'education': {'primary': 'yes',
                                                                                                                                                          'secondary': 'no',
                                                                                                                                                          'tertiary': 'yes',
                                                                                                                                                          'unknown': 'yes'}},
                                                                                                                               'telephone': 'no',
                                                                                                                               'unknown': 'no'}},
                                                                                                       'self-employed': {'marital': {'divorced': 'no',
                                                                                                                                     'married': 'no',
                                                                                                                                     'single': {'education': {'primary': 'no',
                                                                                                                                                              'secondary': 'yes',
                                                                                                                                                              'tertiary': 'yes',
                                                                                                                                                              'unknown': 'yes'}}}},
                                                                                                       'services': {'age>38.0': {'no': 'no',
                                                                                                                                 'yes': 'yes'}},
                                                                                                       'student': 'no',
                                                                                                       'technician': {'marital': {'divorced': 'no',
                                                                                                                                  'married': 'no',
                                                                                                                                  'single': 'yes'}},
                                                                                                       'unemployed': {'education': {'primary': 'no',
                                                                                                                                    'secondary': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                    'yes': 'no'}},
                                                                                                                                    'tertiary': {'housing': {'no': 'yes',
                                                                                                                                                             'yes': 'no'}},
                                                                                                                                    'unknown': 'no'}},
                                                                                                       'unknown': 'no'}},
                                                                                        'yes': {'job': {'admin.': 'yes',
                                                                                                        'blue-collar': 'yes',
                                                                                                        'entrepreneur': 'yes',
                                                                                                        'housemaid': 'yes',
                                                                                                        'management': {'age>38.0': {'no': 'yes',
                                                                                                                                    'yes': 'no'}},
                                                                                                        'retired': {'education': {'primary': 'no',
                                                                                                                                  'secondary': 'yes',
                                                                                                                                  'tertiary': 'yes',
                                                                                                                                  'unknown': 'yes'}},
                                                                                                        'self-employed': 'yes',
                                                                                                        'services': 'yes',
                                                                                                        'student': 'yes',
                                                                                                        'technician': {'contact': {'cellular': 'yes',
                                                                                                                                   'telephone': 'no',
                                                                                                                                   'unknown': 'no'}},
                                                                                                        'unemployed': 'no',
                                                                                                        'unknown': 'yes'}}}},
                                                                   'jan': {'day>16.0': {'no': {'job': {'admin.': 'yes',
                                                                                                       'blue-collar': 'no',
                                                                                                       'entrepreneur': 'yes',
                                                                                                       'housemaid': 'yes',
                                                                                                       'management': 'yes',
                                                                                                       'retired': 'yes',
                                                                                                       'self-employed': 'yes',
                                                                                                       'services': 'yes',
                                                                                                       'student': 'yes',
                                                                                                       'technician': 'yes',
                                                                                                       'unemployed': 'yes',
                                                                                                       'unknown': 'yes'}},
                                                                                        'yes': {'job': {'admin.': {'marital': {'divorced': 'yes',
                                                                                                                               'married': 'no',
                                                                                                                               'single': {'contact': {'cellular': 'no',
                                                                                                                                                      'telephone': 'yes',
                                                                                                                                                      'unknown': 'no'}}}},
                                                                                                        'blue-collar': {'age>38.0': {'no': 'yes',
                                                                                                                                     'yes': 'no'}},
                                                                                                        'entrepreneur': 'no',
                                                                                                        'housemaid': 'no',
                                                                                                        'management': {'campaign>2.0': {'no': 'no',
                                                                                                                                        'yes': {'marital': {'divorced': 'yes',
                                                                                                                                                            'married': 'no',
                                                                                                                                                            'single': 'yes'}}}},
                                                                                                        'retired': {'education': {'primary': 'no',
                                                                                                                                  'secondary': 'yes',
                                                                                                                                  'tertiary': 'no',
                                                                                                                                  'unknown': 'no'}},
                                                                                                        'self-employed': 'no',
                                                                                                        'services': 'no',
                                                                                                        'student': 'no',
                                                                                                        'technician': 'no',
                                                                                                        'unemployed': 'no',
                                                                                                        'unknown': 'no'}}}},
                                                                   'jul': {'job': {'admin.': {'contact': {'cellular': {'marital': {'divorced': {'housing': {'no': 'no',
                                                                                                                                                            'yes': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                      'yes': 'no'}}}},
                                                                                                                                   'married': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                'yes': {'housing': {'no': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                    'yes': 'no'}}}},
                                                                                                                                   'single': {'age>38.0': {'no': {'balance>452.5': {'no': {'housing': {'no': {'day>16.0': {'no': {'loan': {'no': 'yes',
                                                                                                                                                                                                                                           'yes': {'education': {'primary': 'yes',
                                                                                                                                                                                                                                                                 'secondary': {'default': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                 'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                                                                                                                                                 'tertiary': 'yes',
                                                                                                                                                                                                                                                                 'unknown': 'yes'}}}},
                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                    'yes': {'campaign>2.0': {'no': {'education': {'primary': 'yes',
                                                                                                                                                                                                                                  'secondary': {'default': {'no': {'housing': {'no': {'loan': {'no': {'day>16.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                   'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                  'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                          'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                  'tertiary': 'yes',
                                                                                                                                                                                                                                  'unknown': 'yes'}},
                                                                                                                                                                                                             'yes': 'no'}}}},
                                                                                                                                                           'yes': 'no'}}}},
                                                                                                          'telephone': 'yes',
                                                                                                          'unknown': 'no'}},
                                                                                   'blue-collar': {'contact': {'cellular': {'education': {'primary': {'loan': {'no': {'age>38.0': {'no': 'no',
                                                                                                                                                                                   'yes': {'marital': {'divorced': {'housing': {'no': 'no',
                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                       'married': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                                     'yes': {'housing': {'no': {'default': {'no': {'day>16.0': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                        'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                                         'yes': {'default': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                 'yes': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                          'yes': 'no'}}}},
                                                                                                                                                                                                                                                                             'yes': 'no'}}}}}},
                                                                                                                                                                                                       'single': 'yes'}}}},
                                                                                                                                                               'yes': 'no'}},
                                                                                                                                          'secondary': {'marital': {'divorced': 'no',
                                                                                                                                                                    'married': {'default': {'no': {'age>38.0': {'no': {'housing': {'no': 'no',
                                                                                                                                                                                                                                   'yes': {'campaign>2.0': {'no': {'loan': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                'yes': {'balance>452.5': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                          'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                            'yes': {'day>16.0': {'no': {'balance>452.5': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                          'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                 'yes': 'no'}}}},
                                                                                                                                                                                                                                                            'yes': 'no'}}}},
                                                                                                                                                                                                                'yes': {'housing': {'no': {'balance>452.5': {'no': {'day>16.0': {'no': {'loan': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                                                                                                                                                 'yes': 'yes'}},
                                                                                                                                                                                                                                                             'yes': {'day>16.0': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                          'yes': 'no'}},
                                                                                                                                                                                                                                                                                  'yes': 'no'}}}},
                                                                                                                                                                                                                                    'yes': {'day>16.0': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                                                                                                                         'yes': 'no'}}}}}},
                                                                                                                                                                                            'yes': 'no'}},
                                                                                                                                                                    'single': {'housing': {'no': 'no',
                                                                                                                                                                                           'yes': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                     'yes': 'no'}}}}}},
                                                                                                                                          'tertiary': 'no',
                                                                                                                                          'unknown': 'no'}},
                                                                                                               'telephone': 'no',
                                                                                                               'unknown': 'no'}},
                                                                                   'entrepreneur': {'marital': {'divorced': {'housing': {'no': 'yes',
                                                                                                                                         'yes': 'no'}},
                                                                                                                'married': {'education': {'primary': 'no',
                                                                                                                                          'secondary': 'no',
                                                                                                                                          'tertiary': {'housing': {'no': 'yes',
                                                                                                                                                                   'yes': 'no'}},
                                                                                                                                          'unknown': 'no'}},
                                                                                                                'single': 'no'}},
                                                                                   'housemaid': {'balance>452.5': {'no': 'no',
                                                                                                                   'yes': {'marital': {'divorced': 'no',
                                                                                                                                       'married': 'yes',
                                                                                                                                       'single': 'yes'}}}},
                                                                                   'management': {'education': {'primary': 'no',
                                                                                                                'secondary': {'marital': {'divorced': 'no',
                                                                                                                                          'married': 'no',
                                                                                                                                          'single': {'loan': {'no': 'no',
                                                                                                                                                              'yes': 'yes'}}}},
                                                                                                                'tertiary': {'marital': {'divorced': {'age>38.0': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                   'yes': 'no'}},
                                                                                                                                         'married': {'balance>452.5': {'no': 'no',
                                                                                                                                                                       'yes': {'day>16.0': {'no': {'loan': {'no': {'age>38.0': {'no': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                      'yes': {'contact': {'cellular': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                          'telephone': 'yes',
                                                                                                                                                                                                                                                                                          'unknown': 'yes'}}}},
                                                                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                            'yes': 'no'}},
                                                                                                                                                                                            'yes': 'no'}}}},
                                                                                                                                         'single': 'no'}},
                                                                                                                'unknown': {'age>38.0': {'no': 'no',
                                                                                                                                         'yes': 'yes'}}}},
                                                                                   'retired': {'marital': {'divorced': {'education': {'primary': 'no',
                                                                                                                                      'secondary': 'no',
                                                                                                                                      'tertiary': 'yes',
                                                                                                                                      'unknown': 'no'}},
                                                                                                           'married': {'day>16.0': {'no': {'education': {'primary': {'housing': {'no': 'yes',
                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                         'secondary': 'no',
                                                                                                                                                         'tertiary': 'no',
                                                                                                                                                         'unknown': 'no'}},
                                                                                                                                    'yes': 'no'}},
                                                                                                           'single': 'yes'}},
                                                                                   'self-employed': {'marital': {'divorced': {'education': {'primary': 'no',
                                                                                                                                            'secondary': 'yes',
                                                                                                                                            'tertiary': 'no',
                                                                                                                                            'unknown': 'no'}},
                                                                                                                 'married': {'education': {'primary': {'age>38.0': {'no': 'yes',
                                                                                                                                                                    'yes': {'balance>452.5': {'no': {'loan': {'no': 'no',
                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                              'yes': 'no'}}}},
                                                                                                                                           'secondary': 'no',
                                                                                                                                           'tertiary': 'no',
                                                                                                                                           'unknown': 'no'}},
                                                                                                                 'single': 'yes'}},
                                                                                   'services': {'contact': {'cellular': {'marital': {'divorced': {'education': {'primary': 'yes',
                                                                                                                                                                'secondary': {'age>38.0': {'no': {'loan': {'no': 'no',
                                                                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                'tertiary': 'yes',
                                                                                                                                                                'unknown': 'yes'}},
                                                                                                                                     'married': {'day>16.0': {'no': {'loan': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                              'yes': {'age>38.0': {'no': {'education': {'primary': 'no',
                                                                                                                                                                                                        'secondary': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                  'yes': {'balance>452.5': {'no': {'loan': {'no': 'yes',
                                                                                                                                                                                                                                                                            'yes': {'default': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                        'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                'yes': 'yes'}}}},
                                                                                                                                                                                                                                                            'yes': 'no'}}}},
                                                                                                                                                                                                        'tertiary': 'no',
                                                                                                                                                                                                        'unknown': 'no'}},
                                                                                                                                                                                   'yes': 'no'}}}},
                                                                                                                                     'single': {'loan': {'no': 'no',
                                                                                                                                                         'yes': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                   'yes': 'yes'}}}}}},
                                                                                                            'telephone': 'no',
                                                                                                            'unknown': 'yes'}},
                                                                                   'student': 'yes',
                                                                                   'technician': {'marital': {'divorced': 'no',
                                                                                                              'married': {'balance>452.5': {'no': 'no',
                                                                                                                                            'yes': {'education': {'primary': 'no',
                                                                                                                                                                  'secondary': {'loan': {'no': {'contact': {'cellular': {'day>16.0': {'no': 'yes',
                                                                                                                                                                                                                                      'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                                                                           'yes': {'campaign>2.0': {'no': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                          'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                                 'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                    'yes': 'no'}}}}}},
                                                                                                                                                                                                            'telephone': 'no',
                                                                                                                                                                                                            'unknown': 'no'}},
                                                                                                                                                                                         'yes': 'no'}},
                                                                                                                                                                  'tertiary': 'no',
                                                                                                                                                                  'unknown': {'day>16.0': {'no': 'no',
                                                                                                                                                                                           'yes': 'yes'}}}}}},
                                                                                                              'single': {'contact': {'cellular': {'default': {'no': {'loan': {'no': {'education': {'primary': 'no',
                                                                                                                                                                                                   'secondary': 'no',
                                                                                                                                                                                                   'tertiary': {'housing': {'no': 'yes',
                                                                                                                                                                                                                            'yes': 'no'}},
                                                                                                                                                                                                   'unknown': 'no'}},
                                                                                                                                                                              'yes': {'education': {'primary': 'yes',
                                                                                                                                                                                                    'secondary': 'yes',
                                                                                                                                                                                                    'tertiary': {'balance>452.5': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                   'yes': 'no'}},
                                                                                                                                                                                                    'unknown': 'yes'}}}},
                                                                                                                                                              'yes': 'no'}},
                                                                                                                                     'telephone': 'no',
                                                                                                                                     'unknown': {'education': {'primary': 'yes',
                                                                                                                                                               'secondary': 'no',
                                                                                                                                                               'tertiary': 'yes',
                                                                                                                                                               'unknown': 'yes'}}}}}},
                                                                                   'unemployed': {'education': {'primary': 'no',
                                                                                                                'secondary': {'day>16.0': {'no': {'age>38.0': {'no': 'no',
                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                           'yes': 'no'}},
                                                                                                                'tertiary': 'yes',
                                                                                                                'unknown': {'balance>452.5': {'no': 'no',
                                                                                                                                              'yes': 'yes'}}}},
                                                                                   'unknown': 'no'}},
                                                                   'jun': {'contact': {'cellular': {'job': {'admin.': 'yes',
                                                                                                            'blue-collar': 'yes',
                                                                                                            'entrepreneur': 'yes',
                                                                                                            'housemaid': 'yes',
                                                                                                            'management': {'day>16.0': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                                           'married': {'age>38.0': {'no': 'no',
                                                                                                                                                                                    'yes': {'housing': {'no': {'education': {'primary': 'yes',
                                                                                                                                                                                                                             'secondary': 'yes',
                                                                                                                                                                                                                             'tertiary': {'default': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                                                                               'yes': {'loan': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                'yes': 'yes'}}}},
                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                             'unknown': 'yes'}},
                                                                                                                                                                                                        'yes': 'no'}}}},
                                                                                                                                                           'single': {'loan': {'no': 'yes',
                                                                                                                                                                               'yes': 'no'}}}},
                                                                                                                                        'yes': 'yes'}},
                                                                                                            'retired': {'education': {'primary': {'balance>452.5': {'no': 'no',
                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                      'secondary': 'yes',
                                                                                                                                      'tertiary': 'no',
                                                                                                                                      'unknown': 'yes'}},
                                                                                                            'self-employed': 'yes',
                                                                                                            'services': {'marital': {'divorced': 'yes',
                                                                                                                                     'married': 'yes',
                                                                                                                                     'single': 'no'}},
                                                                                                            'student': 'yes',
                                                                                                            'technician': {'balance>452.5': {'no': 'yes',
                                                                                                                                             'yes': 'no'}},
                                                                                                            'unemployed': {'marital': {'divorced': 'yes',
                                                                                                                                       'married': 'no',
                                                                                                                                       'single': 'yes'}},
                                                                                                            'unknown': 'yes'}},
                                                                                       'telephone': {'job': {'admin.': 'yes',
                                                                                                             'blue-collar': 'yes',
                                                                                                             'entrepreneur': 'yes',
                                                                                                             'housemaid': 'yes',
                                                                                                             'management': 'yes',
                                                                                                             'retired': 'yes',
                                                                                                             'self-employed': 'yes',
                                                                                                             'services': 'yes',
                                                                                                             'student': 'yes',
                                                                                                             'technician': 'no',
                                                                                                             'unemployed': 'yes',
                                                                                                             'unknown': 'yes'}},
                                                                                       'unknown': {'job': {'admin.': {'balance>452.5': {'no': {'age>38.0': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                                                               'married': {'loan': {'no': 'yes',
                                                                                                                                                                                                    'yes': 'no'}},
                                                                                                                                                                               'single': {'housing': {'no': 'yes',
                                                                                                                                                                                                      'yes': 'no'}}}},
                                                                                                                                                            'yes': 'no'}},
                                                                                                                                        'yes': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                 'yes': {'housing': {'no': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                     'yes': 'no'}}}}}},
                                                                                                           'blue-collar': {'age>38.0': {'no': {'default': {'no': {'loan': {'no': {'housing': {'no': 'no',
                                                                                                                                                                                              'yes': {'education': {'primary': {'campaign>2.0': {'no': {'marital': {'divorced': 'yes',
                                                                                                                                                                                                                                                                    'married': {'day>16.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                             'yes': 'no'}},
                                                                                                                                                                                                                                                                    'single': 'yes'}},
                                                                                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                                                                                    'secondary': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                                   'yes': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                                                                             'yes': {'marital': {'divorced': 'no',
                                                                                                                                                                                                                                                                                                 'married': {'day>16.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                          'yes': 'no'}},
                                                                                                                                                                                                                                                                                                 'single': 'no'}}}}}},
                                                                                                                                                                                                                    'tertiary': 'no',
                                                                                                                                                                                                                    'unknown': 'no'}}}},
                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                        'yes': {'marital': {'divorced': {'education': {'primary': {'default': {'no': 'yes',
                                                                                                                                                                                                               'yes': 'no'}},
                                                                                                                                                                                       'secondary': 'no',
                                                                                                                                                                                       'tertiary': 'no',
                                                                                                                                                                                       'unknown': 'no'}},
                                                                                                                                                            'married': 'no',
                                                                                                                                                            'single': 'no'}}}},
                                                                                                           'entrepreneur': {'marital': {'divorced': 'no',
                                                                                                                                        'married': {'education': {'primary': 'no',
                                                                                                                                                                  'secondary': 'no',
                                                                                                                                                                  'tertiary': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                 'yes': 'yes'}},
                                                                                                                                                                  'unknown': 'no'}},
                                                                                                                                        'single': 'no'}},
                                                                                                           'housemaid': 'no',
                                                                                                           'management': {'campaign>2.0': {'no': {'marital': {'divorced': 'no',
                                                                                                                                                              'married': {'housing': {'no': {'balance>452.5': {'no': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                                                   'yes': {'education': {'primary': 'no',
                                                                                                                                                                                                                                                         'secondary': 'no',
                                                                                                                                                                                                                                                         'tertiary': {'default': {'no': {'loan': {'no': {'day>16.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                    'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                            'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                                                                                                  'yes': 'no'}},
                                                                                                                                                                                                                                                                                  'yes': 'no'}},
                                                                                                                                                                                                                                                         'unknown': 'no'}}}},
                                                                                                                                                                                                               'yes': 'no'}},
                                                                                                                                                                                      'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                           'yes': 'yes'}}}},
                                                                                                                                                              'single': 'no'}},
                                                                                                                                           'yes': 'no'}},
                                                                                                           'retired': {'education': {'primary': 'no',
                                                                                                                                     'secondary': {'marital': {'divorced': 'no',
                                                                                                                                                               'married': {'housing': {'no': {'balance>452.5': {'no': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                                                                    'yes': {'default': {'no': {'loan': {'no': {'day>16.0': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                  'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                          'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                        'yes': 'yes'}}}},
                                                                                                                                                                                                                'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                                                     'yes': {'default': {'no': {'loan': {'no': {'day>16.0': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                   'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                                                                                                                                                                             'yes': 'no'}},
                                                                                                                                                                                                                                                                         'yes': 'no'}},
                                                                                                                                                                                                                                                         'yes': 'no'}}}}}},
                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                               'single': 'no'}},
                                                                                                                                     'tertiary': 'no',
                                                                                                                                     'unknown': 'no'}},
                                                                                                           'self-employed': {'default': {'no': 'no',
                                                                                                                                         'yes': {'marital': {'divorced': 'no',
                                                                                                                                                             'married': 'yes',
                                                                                                                                                             'single': 'yes'}}}},
                                                                                                           'services': {'age>38.0': {'no': {'balance>452.5': {'no': {'loan': {'no': {'education': {'primary': 'yes',
                                                                                                                                                                                                   'secondary': {'marital': {'divorced': {'default': {'no': {'housing': {'no': {'day>16.0': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                                                                                                                             'yes': 'yes'}},
                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                             'married': 'yes',
                                                                                                                                                                                                                             'single': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                       'yes': {'day>16.0': {'no': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                    'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                           'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                            'yes': 'yes'}}}},
                                                                                                                                                                                                                                                    'yes': 'yes'}}}},
                                                                                                                                                                                                   'tertiary': 'no',
                                                                                                                                                                                                   'unknown': 'yes'}},
                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                              'yes': 'no'}},
                                                                                                                                     'yes': 'no'}},
                                                                                                           'student': 'no',
                                                                                                           'technician': {'marital': {'divorced': {'age>38.0': {'no': 'yes',
                                                                                                                                                                'yes': 'no'}},
                                                                                                                                      'married': 'no',
                                                                                                                                      'single': {'balance>452.5': {'no': 'no',
                                                                                                                                                                   'yes': 'yes'}}}},
                                                                                                           'unemployed': 'no',
                                                                                                           'unknown': 'no'}}}},
                                                                   'mar': {'marital': {'divorced': 'no',
                                                                                       'married': {'day>16.0': {'no': 'yes',
                                                                                                                'yes': {'job': {'admin.': 'yes',
                                                                                                                                'blue-collar': 'no',
                                                                                                                                'entrepreneur': 'no',
                                                                                                                                'housemaid': 'no',
                                                                                                                                'management': 'no',
                                                                                                                                'retired': 'no',
                                                                                                                                'self-employed': 'no',
                                                                                                                                'services': 'no',
                                                                                                                                'student': 'no',
                                                                                                                                'technician': 'no',
                                                                                                                                'unemployed': 'no',
                                                                                                                                'unknown': 'no'}}}},
                                                                                       'single': 'yes'}},
                                                                   'may': {'contact': {'cellular': {'job': {'admin.': {'education': {'primary': 'no',
                                                                                                                                     'secondary': {'age>38.0': {'no': {'marital': {'divorced': 'no',
                                                                                                                                                                                   'married': 'no',
                                                                                                                                                                                   'single': {'loan': {'no': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                                'yes': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                                     'yes': 'yes'}}}},
                                                                                                                                                                                                       'yes': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                                 'yes': 'no'}}}}}},
                                                                                                                                                                'yes': {'marital': {'divorced': 'yes',
                                                                                                                                                                                    'married': {'loan': {'no': 'no',
                                                                                                                                                                                                         'yes': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                  'yes': 'no'}}}},
                                                                                                                                                                                    'single': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                 'yes': 'yes'}}}}}},
                                                                                                                                     'tertiary': 'yes',
                                                                                                                                     'unknown': 'no'}},
                                                                                                            'blue-collar': {'marital': {'divorced': {'education': {'primary': {'loan': {'no': {'age>38.0': {'no': {'housing': {'no': 'no',
                                                                                                                                                                                                                               'yes': {'campaign>2.0': {'no': 'yes',
                                                                                                                                                                                                                                                        'yes': 'no'}}}},
                                                                                                                                                                                                            'yes': 'no'}},
                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                   'secondary': 'no',
                                                                                                                                                                   'tertiary': 'no',
                                                                                                                                                                   'unknown': 'no'}},
                                                                                                                                        'married': {'balance>452.5': {'no': 'no',
                                                                                                                                                                      'yes': {'education': {'primary': {'age>38.0': {'no': {'day>16.0': {'no': 'yes',
                                                                                                                                                                                                                                         'yes': 'no'}},
                                                                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                                                                            'secondary': 'no',
                                                                                                                                                                                            'tertiary': 'no',
                                                                                                                                                                                            'unknown': 'no'}}}},
                                                                                                                                        'single': {'education': {'primary': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                               'yes': 'no'}},
                                                                                                                                                                 'secondary': {'day>16.0': {'no': 'no',
                                                                                                                                                                                            'yes': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                      'yes': 'yes'}}}},
                                                                                                                                                                 'tertiary': 'yes',
                                                                                                                                                                 'unknown': 'no'}}}},
                                                                                                            'entrepreneur': 'no',
                                                                                                            'housemaid': 'no',
                                                                                                            'management': {'housing': {'no': {'balance>452.5': {'no': {'age>38.0': {'no': 'no',
                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                       'yes': {'day>16.0': {'no': {'campaign>2.0': {'no': {'education': {'primary': 'no',
                                                                                                                                                                                                         'secondary': 'no',
                                                                                                                                                                                                         'tertiary': {'marital': {'divorced': 'yes',
                                                                                                                                                                                                                                  'married': {'age>38.0': {'no': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                                                                                    'yes': {'loan': {'no': {'default': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                     'yes': {'default': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                         'yes': 'yes'}}}}}},
                                                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                                                  'single': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                                                                                               'yes': {'age>38.0': {'no': {'default': {'no': {'loan': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                             'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                                                                                                                                                    'yes': 'no'}}}}}},
                                                                                                                                                                                                         'unknown': 'no'}},
                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                            'yes': 'no'}}}},
                                                                                                            'retired': {'education': {'primary': 'no',
                                                                                                                                      'secondary': 'no',
                                                                                                                                      'tertiary': {'balance>452.5': {'no': 'no',
                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                      'unknown': 'no'}},
                                                                                                            'self-employed': {'marital': {'divorced': 'yes',
                                                                                                                                          'married': 'no',
                                                                                                                                          'single': {'balance>452.5': {'no': 'no',
                                                                                                                                                                       'yes': 'yes'}}}},
                                                                                                            'services': {'marital': {'divorced': 'no',
                                                                                                                                     'married': {'loan': {'no': {'campaign>2.0': {'no': {'balance>452.5': {'no': {'age>38.0': {'no': {'education': {'primary': 'yes',
                                                                                                                                                                                                                                                    'secondary': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                                 'yes': {'day>16.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                    'tertiary': 'yes',
                                                                                                                                                                                                                                                    'unknown': 'yes'}},
                                                                                                                                                                                                                               'yes': 'no'}},
                                                                                                                                                                                                           'yes': {'education': {'primary': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                                                                                          'yes': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                                 'yes': {'day>16.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                    'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                            'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                              'yes': 'yes'}}}},
                                                                                                                                                                                                                                 'secondary': 'no',
                                                                                                                                                                                                                                 'tertiary': 'no',
                                                                                                                                                                                                                                 'unknown': 'no'}}}},
                                                                                                                                                                                  'yes': 'yes'}},
                                                                                                                                                          'yes': 'no'}},
                                                                                                                                     'single': {'age>38.0': {'no': {'day>16.0': {'no': {'campaign>2.0': {'no': {'balance>452.5': {'no': {'loan': {'no': 'yes',
                                                                                                                                                                                                                                                  'yes': 'no'}},
                                                                                                                                                                                                                                  'yes': 'no'}},
                                                                                                                                                                                                         'yes': 'no'}},
                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                             'yes': 'no'}}}},
                                                                                                            'student': {'education': {'primary': 'no',
                                                                                                                                      'secondary': 'no',
                                                                                                                                      'tertiary': 'yes',
                                                                                                                                      'unknown': 'no'}},
                                                                                                            'technician': {'marital': {'divorced': {'education': {'primary': 'yes',
                                                                                                                                                                  'secondary': 'yes',
                                                                                                                                                                  'tertiary': 'no',
                                                                                                                                                                  'unknown': 'yes'}},
                                                                                                                                       'married': 'no',
                                                                                                                                       'single': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                   'yes': {'balance>452.5': {'no': 'no',
                                                                                                                                                                                             'yes': 'yes'}}}}}},
                                                                                                            'unemployed': 'no',
                                                                                                            'unknown': 'no'}},
                                                                                       'telephone': {'job': {'admin.': 'no',
                                                                                                             'blue-collar': 'no',
                                                                                                             'entrepreneur': 'no',
                                                                                                             'housemaid': 'no',
                                                                                                             'management': 'no',
                                                                                                             'retired': 'no',
                                                                                                             'self-employed': 'no',
                                                                                                             'services': 'no',
                                                                                                             'student': 'no',
                                                                                                             'technician': {'age>38.0': {'no': 'no',
                                                                                                                                         'yes': 'yes'}},
                                                                                                             'unemployed': 'yes',
                                                                                                             'unknown': 'yes'}},
                                                                                       'unknown': {'balance>452.5': {'no': {'job': {'admin.': {'day>16.0': {'no': 'no',
                                                                                                                                                            'yes': {'campaign>2.0': {'no': {'loan': {'no': {'marital': {'divorced': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                                                                                  'yes': 'no'}},
                                                                                                                                                                                                                        'married': 'no',
                                                                                                                                                                                                                        'single': 'no'}},
                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                     'yes': 'no'}}}},
                                                                                                                                    'blue-collar': {'education': {'primary': 'no',
                                                                                                                                                                  'secondary': {'day>16.0': {'no': {'marital': {'divorced': 'no',
                                                                                                                                                                                                                'married': 'no',
                                                                                                                                                                                                                'single': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                                                        'yes': {'loan': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                                                                 'yes': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                        'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                     'yes': 'yes'}}}},
                                                                                                                                                                                                                                                         'yes': 'no'}}}}}},
                                                                                                                                                                                             'yes': {'marital': {'divorced': {'age>38.0': {'no': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                 'yes': {'loan': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                          'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                  'yes': 'yes'}}}},
                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                                 'married': {'loan': {'no': {'campaign>2.0': {'no': {'age>38.0': {'no': {'default': {'no': {'housing': {'no': 'no',
                                                                                                                                                                                                                                                                                                                        'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'no'}}}},
                                                                                                                                                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                                                                                                                                                                  'yes': {'default': {'no': {'housing': {'no': 'no',
                                                                                                                                                                                                                                                                                                                         'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'no'}}}},
                                                                                                                                                                                                                                                                                                      'yes': 'no'}}}},
                                                                                                                                                                                                                                                              'yes': {'age>38.0': {'no': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                         'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                   'yes': 'no'}}}},
                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                 'single': 'no'}}}},
                                                                                                                                                                  'tertiary': 'no',
                                                                                                                                                                  'unknown': 'no'}},
                                                                                                                                    'entrepreneur': {'marital': {'divorced': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                 'married': 'no',
                                                                                                                                                                 'single': 'no'}},
                                                                                                                                    'housemaid': 'no',
                                                                                                                                    'management': 'no',
                                                                                                                                    'retired': 'no',
                                                                                                                                    'self-employed': 'no',
                                                                                                                                    'services': {'education': {'primary': 'no',
                                                                                                                                                               'secondary': 'no',
                                                                                                                                                               'tertiary': 'no',
                                                                                                                                                               'unknown': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                        'yes': 'no'}}}},
                                                                                                                                    'student': 'no',
                                                                                                                                    'technician': 'no',
                                                                                                                                    'unemployed': 'no',
                                                                                                                                    'unknown': 'no'}},
                                                                                                                     'yes': {'job': {'admin.': {'marital': {'divorced': 'no',
                                                                                                                                                            'married': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                         'yes': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                                              'yes': 'no'}}}},
                                                                                                                                                            'single': 'no'}},
                                                                                                                                     'blue-collar': {'education': {'primary': {'marital': {'divorced': 'no',
                                                                                                                                                                                           'married': {'loan': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                    'yes': {'age>38.0': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                                                                 'yes': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                        'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                     'yes': 'yes'}}}},
                                                                                                                                                                                                                                                         'yes': {'campaign>2.0': {'no': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                        'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                                                                                                                  'yes': 'no'}}}}}},
                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                           'single': {'age>38.0': {'no': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                           'yes': 'yes'}},
                                                                                                                                                                                                                   'yes': 'yes'}}}},
                                                                                                                                                                   'secondary': {'age>38.0': {'no': {'loan': {'no': {'housing': {'no': 'no',
                                                                                                                                                                                                                                 'yes': {'marital': {'divorced': 'no',
                                                                                                                                                                                                                                                     'married': {'campaign>2.0': {'no': 'no',
                                                                                                                                                                                                                                                                                  'yes': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                       'yes': {'default': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                         'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                           'yes': 'no'}}}}}},
                                                                                                                                                                                                                                                     'single': {'day>16.0': {'no': {'default': {'no': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                        'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                'yes': 'yes'}},
                                                                                                                                                                                                                                                                             'yes': 'no'}}}}}},
                                                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                                   'tertiary': {'age>38.0': {'no': 'no',
                                                                                                                                                                                             'yes': {'marital': {'divorced': 'yes',
                                                                                                                                                                                                                 'married': 'no',
                                                                                                                                                                                                                 'single': 'yes'}}}},
                                                                                                                                                                   'unknown': 'no'}},
                                                                                                                                     'entrepreneur': {'education': {'primary': 'no',
                                                                                                                                                                    'secondary': 'no',
                                                                                                                                                                    'tertiary': 'no',
                                                                                                                                                                    'unknown': 'yes'}},
                                                                                                                                     'housemaid': 'no',
                                                                                                                                     'management': {'marital': {'divorced': {'loan': {'no': {'campaign>2.0': {'no': {'day>16.0': {'no': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                                  'yes': 'no'}},
                                                                                                                                                                                                              'yes': 'yes'}},
                                                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                                                'married': 'no',
                                                                                                                                                                'single': {'day>16.0': {'no': 'no',
                                                                                                                                                                                        'yes': {'campaign>2.0': {'no': {'age>38.0': {'no': {'education': {'primary': 'yes',
                                                                                                                                                                                                                                                          'secondary': 'yes',
                                                                                                                                                                                                                                                          'tertiary': {'default': {'no': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                                                                      'yes': {'loan': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                             'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}}}},
                                                                                                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                                                                                          'unknown': 'yes'}},
                                                                                                                                                                                                                                     'yes': 'yes'}},
                                                                                                                                                                                                                 'yes': 'no'}}}}}},
                                                                                                                                     'retired': 'no',
                                                                                                                                     'self-employed': {'age>38.0': {'no': {'housing': {'no': 'no',
                                                                                                                                                                                       'yes': {'marital': {'divorced': 'no',
                                                                                                                                                                                                           'married': 'no',
                                                                                                                                                                                                           'single': {'education': {'primary': 'no',
                                                                                                                                                                                                                                    'secondary': 'no',
                                                                                                                                                                                                                                    'tertiary': {'default': {'no': {'loan': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                 'yes': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                                'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                          'yes': 'no'}}}},
                                                                                                                                                                                                                                                                             'yes': 'no'}},
                                                                                                                                                                                                                                                             'yes': 'no'}},
                                                                                                                                                                                                                                    'unknown': 'no'}}}}}},
                                                                                                                                                                    'yes': 'no'}},
                                                                                                                                     'services': 'no',
                                                                                                                                     'student': 'no',
                                                                                                                                     'technician': {'marital': {'divorced': {'age>38.0': {'no': 'yes',
                                                                                                                                                                                          'yes': {'education': {'primary': 'yes',
                                                                                                                                                                                                                'secondary': 'no',
                                                                                                                                                                                                                'tertiary': 'no',
                                                                                                                                                                                                                'unknown': 'no'}}}},
                                                                                                                                                                'married': 'no',
                                                                                                                                                                'single': {'loan': {'no': {'age>38.0': {'no': {'education': {'primary': 'yes',
                                                                                                                                                                                                                             'secondary': 'yes',
                                                                                                                                                                                                                             'tertiary': {'campaign>2.0': {'no': {'day>16.0': {'no': 'yes',
                                                                                                                                                                                                                                                                               'yes': {'default': {'no': {'housing': {'no': 'no',
                                                                                                                                                                                                                                                                                                                      'yes': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                     'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                             'yes': 'no'}}}},
                                                                                                                                                                                                                                                                                                   'yes': 'no'}}}},
                                                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                                             'unknown': 'yes'}},
                                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                                                    'yes': 'no'}}}},
                                                                                                                                     'unemployed': 'no',
                                                                                                                                     'unknown': 'no'}}}}}},
                                                                   'nov': {'job': {'admin.': {'housing': {'no': {'age>38.0': {'no': 'no',
                                                                                                                              'yes': {'contact': {'cellular': 'no',
                                                                                                                                                  'telephone': 'yes',
                                                                                                                                                  'unknown': 'yes'}}}},
                                                                                                          'yes': 'no'}},
                                                                                   'blue-collar': 'no',
                                                                                   'entrepreneur': {'education': {'primary': 'yes',
                                                                                                                  'secondary': 'no',
                                                                                                                  'tertiary': {'age>38.0': {'no': 'no',
                                                                                                                                            'yes': {'marital': {'divorced': 'no',
                                                                                                                                                                'married': 'yes',
                                                                                                                                                                'single': 'yes'}}}},
                                                                                                                  'unknown': 'no'}},
                                                                                   'housemaid': 'no',
                                                                                   'management': {'day>16.0': {'no': {'age>38.0': {'no': {'housing': {'no': 'no',
                                                                                                                                                      'yes': 'yes'}},
                                                                                                                                   'yes': 'yes'}},
                                                                                                               'yes': {'marital': {'divorced': {'education': {'primary': 'no',
                                                                                                                                                              'secondary': 'yes',
                                                                                                                                                              'tertiary': {'age>38.0': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                 'yes': 'no'}},
                                                                                                                                                                                        'yes': 'no'}},
                                                                                                                                                              'unknown': 'no'}},
                                                                                                                                   'married': {'campaign>2.0': {'no': {'housing': {'no': {'loan': {'no': {'balance>452.5': {'no': {'education': {'primary': 'no',
                                                                                                                                                                                                                                                 'secondary': 'no',
                                                                                                                                                                                                                                                 'tertiary': {'age>38.0': {'no': 'no',
                                                                                                                                                                                                                                                                           'yes': {'default': {'no': {'contact': {'cellular': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'no',
                                                                                                                                                                                                                                                                                                                                                                      'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                                              'yes': 'no'}},
                                                                                                                                                                                                                                                                                                                  'telephone': 'no',
                                                                                                                                                                                                                                                                                                                  'unknown': 'no'}},
                                                                                                                                                                                                                                                                                               'yes': 'no'}}}},
                                                                                                                                                                                                                                                 'unknown': 'no'}},
                                                                                                                                                                                                                            'yes': 'no'}},
                                                                                                                                                                                                   'yes': 'yes'}},
                                                                                                                                                                                   'yes': 'no'}},
                                                                                                                                                                'yes': {'default': {'no': {'age>38.0': {'no': {'housing': {'no': {'loan': {'no': 'yes',
                                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                                           'yes': 'no'}},
                                                                                                                                                                                                        'yes': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                                  'yes': {'housing': {'no': 'no',
                                                                                                                                                                                                                                                      'yes': 'yes'}}}}}},
                                                                                                                                                                                    'yes': 'no'}}}},
                                                                                                                                   'single': 'no'}}}},
                                                                                   'retired': {'education': {'primary': 'no',
                                                                                                             'secondary': 'no',
                                                                                                             'tertiary': {'marital': {'divorced': 'yes',
                                                                                                                                      'married': 'yes',
                                                                                                                                      'single': 'no'}},
                                                                                                             'unknown': 'no'}},
                                                                                   'self-employed': {'marital': {'divorced': 'no',
                                                                                                                 'married': 'no',
                                                                                                                 'single': {'education': {'primary': 'yes',
                                                                                                                                          'secondary': 'yes',
                                                                                                                                          'tertiary': 'no',
                                                                                                                                          'unknown': 'yes'}}}},
                                                                                   'services': {'education': {'primary': 'no',
                                                                                                              'secondary': {'day>16.0': {'no': 'yes',
                                                                                                                                         'yes': {'age>38.0': {'no': 'no',
                                                                                                                                                              'yes': {'housing': {'no': {'loan': {'no': 'yes',
                                                                                                                                                                                                  'yes': 'no'}},
                                                                                                                                                                                  'yes': 'no'}}}}}},
                                                                                                              'tertiary': 'no',
                                                                                                              'unknown': 'yes'}},
                                                                                   'student': 'no',
                                                                                   'technician': {'marital': {'divorced': 'no',
                                                                                                              'married': 'no',
                                                                                                              'single': {'age>38.0': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                                          'yes': {'education': {'primary': 'yes',
                                                                                                                                                                                'secondary': {'default': {'no': {'balance>452.5': {'no': 'yes',
                                                                                                                                                                                                                                   'yes': {'housing': {'no': 'yes',
                                                                                                                                                                                                                                                       'yes': {'loan': {'no': {'contact': {'cellular': {'campaign>2.0': {'no': {'pdays>-1.0': {'no': {'previous>0.0': {'no': 'yes',
                                                                                                                                                                                                                                                                                                                                                                       'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                                               'yes': 'yes'}},
                                                                                                                                                                                                                                                                                                                         'yes': 'yes'}},
                                                                                                                                                                                                                                                                                           'telephone': 'yes',
                                                                                                                                                                                                                                                                                           'unknown': 'yes'}},
                                                                                                                                                                                                                                                                        'yes': 'yes'}}}}}},
                                                                                                                                                                                                          'yes': 'yes'}},
                                                                                                                                                                                'tertiary': 'yes',
                                                                                                                                                                                'unknown': 'yes'}}}},
                                                                                                                                      'yes': 'no'}}}},
                                                                                   'unemployed': 'no',
                                                                                   'unknown': 'no'}},
                                                                   'oct': {'job': {'admin.': {'age>38.0': {'no': 'yes',
                                                                                                           'yes': 'no'}},
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'yes',
                                                                                   'management': {'marital': {'divorced': 'yes',
                                                                                                              'married': 'no',
                                                                                                              'single': {'balance>452.5': {'no': 'yes',
                                                                                                                                           'yes': {'day>16.0': {'no': 'yes',
                                                                                                                                                                'yes': 'no'}}}}}},
                                                                                   'retired': {'marital': {'divorced': 'no',
                                                                                                           'married': 'no',
                                                                                                           'single': 'yes'}},
                                                                                   'self-employed': 'yes',
                                                                                   'services': 'yes',
                                                                                   'student': 'yes',
                                                                                   'technician': 'yes',
                                                                                   'unemployed': 'yes',
                                                                                   'unknown': 'yes'}},
                                                                   'sep': {'job': {'admin.': {'marital': {'divorced': 'no',
                                                                                                          'married': 'no',
                                                                                                          'single': {'education': {'primary': 'yes',
                                                                                                                                   'secondary': 'yes',
                                                                                                                                   'tertiary': 'yes',
                                                                                                                                   'unknown': 'no'}}}},
                                                                                   'blue-collar': 'yes',
                                                                                   'entrepreneur': 'yes',
                                                                                   'housemaid': 'yes',
                                                                                   'management': {'age>38.0': {'no': {'day>16.0': {'no': 'no',
                                                                                                                                   'yes': 'yes'}},
                                                                                                               'yes': 'yes'}},
                                                                                   'retired': 'yes',
                                                                                   'self-employed': 'yes',
                                                                                   'services': 'no',
                                                                                   'student': 'no',
                                                                                   'technician': 'yes',
                                                                                   'unemployed': 'no',
                                                                                   'unknown': 'yes'}}}}}}}}



```python
Our_prediction = df_train_new.apply(prediction, axis=1, args = [trained_tree])
np.where(df_train_new['y'] == Our_prediction,1,0).mean()
```




    0.9868




```python
Our_prediction = df_test_new.apply(prediction, axis=1, args = [trained_tree])
np.where(df_test_new['y'] == Our_prediction,1,0).mean()
```




    0.8394




```python
# dealing with missing values

df_train_correction = pd.DataFrame()
df_test_correction = pd.DataFrame()
for name in list(df_train_new.keys()):
    m = df_train_new[name].value_counts().idxmax(axis = 0)
    print('Majority in column %s at trauning is ' %name, m)
    df_train_correction[name] = np.where(df_train_new[name] == 'unknown', m, df_train_new[name])
    n = df_test_new[name].value_counts().idxmax(axis = 0)
    print('Majority in column %s at test data is ' %name, n)
    df_test_correction[name] = np.where(df_test_new[name] == 'unknown', n, df_test_new[name])
```

    Majority in column age>38.0 at trauning is  no
    Majority in column age>38.0 at test data is  yes
    Majority in column job at trauning is  blue-collar
    Majority in column job at test data is  blue-collar
    Majority in column marital at trauning is  married
    Majority in column marital at test data is  married
    Majority in column education at trauning is  secondary
    Majority in column education at test data is  secondary
    Majority in column default at trauning is  no
    Majority in column default at test data is  no
    Majority in column balance>452.5 at trauning is  yes
    Majority in column balance>452.5 at test data is  yes
    Majority in column housing at trauning is  yes
    Majority in column housing at test data is  yes
    Majority in column loan at trauning is  no
    Majority in column loan at test data is  no
    Majority in column contact at trauning is  cellular
    Majority in column contact at test data is  cellular
    Majority in column day>16.0 at trauning is  no
    Majority in column day>16.0 at test data is  no
    Majority in column month at trauning is  may
    Majority in column month at test data is  may
    Majority in column duration>180.0 at trauning is  no
    Majority in column duration>180.0 at test data is  no
    Majority in column campaign>2.0 at trauning is  no
    Majority in column campaign>2.0 at test data is  no
    Majority in column pdays>-1.0 at trauning is  no
    Majority in column pdays>-1.0 at test data is  no
    Majority in column previous>0.0 at trauning is  no
    Majority in column previous>0.0 at test data is  no
    Majority in column poutcome at trauning is  unknown
    Majority in column poutcome at test data is  unknown
    Majority in column y at trauning is  no
    Majority in column y at test data is  no



```python
df_train_correction.loc[[1806]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age&gt;38.0</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance&gt;452.5</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day&gt;16.0</th>
      <th>month</th>
      <th>duration&gt;180.0</th>
      <th>campaign&gt;2.0</th>
      <th>pdays&gt;-1.0</th>
      <th>previous&gt;0.0</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1806</td>
      <td>no</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>yes</td>
      <td>jan</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test_correction.loc[[1806]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age&gt;38.0</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance&gt;452.5</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day&gt;16.0</th>
      <th>month</th>
      <th>duration&gt;180.0</th>
      <th>campaign&gt;2.0</th>
      <th>pdays&gt;-1.0</th>
      <th>previous&gt;0.0</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1806</td>
      <td>no</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>yes</td>
      <td>may</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train_correction.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age&gt;38.0</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance&gt;452.5</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day&gt;16.0</th>
      <th>month</th>
      <th>duration&gt;180.0</th>
      <th>campaign&gt;2.0</th>
      <th>pdays&gt;-1.0</th>
      <th>previous&gt;0.0</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>yes</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>no</td>
      <td>may</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>1</td>
      <td>yes</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>no</td>
      <td>feb</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>2</td>
      <td>yes</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>cellular</td>
      <td>yes</td>
      <td>aug</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>success</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>3</td>
      <td>yes</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>no</td>
      <td>jul</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <td>4</td>
      <td>no</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>no</td>
      <td>apr</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
tabel_train_data = {}
tabel_test_data = {}
for ptype in ["Entropy","Gini_Index","Majority_Error"]:
    tabel_train_data[ptype]= np.zeros(16)
    tabel_test_data[ptype]= np.zeros(16)
    for i in range(16):
        trained_tree = build_tree(df_train_new, 'y', depth = i+1, type_tree = ptype)

        Our_prediction_train = df_train_new.apply(prediction, axis=1, args = [trained_tree])
        tabel_train_data[ptype][i]= 1- np.where(df_train_new['y'] == Our_prediction_train,1,0).mean()
        
        Our_prediction_test = df_test_new.apply(prediction, axis=1, args = [trained_tree])
        tabel_test_data[ptype][i]= 1- np.where(df_test_new['y'] == Our_prediction_test,1,0).mean()
        print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15



```python
tabel_df = pd.DataFrame.from_dict(tabel_train_data)
tabel_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entropy</th>
      <th>Gini_Index</th>
      <th>Majority_Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.1192</td>
      <td>0.1088</td>
      <td>0.1088</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.1060</td>
      <td>0.1042</td>
      <td>0.1042</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.1006</td>
      <td>0.0936</td>
      <td>0.0966</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0800</td>
      <td>0.0754</td>
      <td>0.0840</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0624</td>
      <td>0.0604</td>
      <td>0.0686</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.0480</td>
      <td>0.0484</td>
      <td>0.0626</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.0372</td>
      <td>0.0364</td>
      <td>0.0584</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.0292</td>
      <td>0.0268</td>
      <td>0.0528</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.0222</td>
      <td>0.0216</td>
      <td>0.0488</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.0182</td>
      <td>0.0174</td>
      <td>0.0438</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.0150</td>
      <td>0.0140</td>
      <td>0.0384</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.0136</td>
      <td>0.0136</td>
      <td>0.0316</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.0132</td>
      <td>0.0132</td>
      <td>0.0256</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.0132</td>
      <td>0.0132</td>
      <td>0.0198</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.0132</td>
      <td>0.0132</td>
      <td>0.0166</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.0132</td>
      <td>0.0132</td>
      <td>0.0132</td>
    </tr>
  </tbody>
</table>
</div>




```python
tabel_df_test = pd.DataFrame.from_dict(tabel_test_data)
tabel_df_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entropy</th>
      <th>Gini_Index</th>
      <th>Majority_Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.1248</td>
      <td>0.1166</td>
      <td>0.1166</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.1114</td>
      <td>0.1088</td>
      <td>0.1088</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.1074</td>
      <td>0.1154</td>
      <td>0.1146</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.1198</td>
      <td>0.1222</td>
      <td>0.1176</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.1282</td>
      <td>0.1338</td>
      <td>0.1210</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.1346</td>
      <td>0.1462</td>
      <td>0.1268</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.1412</td>
      <td>0.1520</td>
      <td>0.1282</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.1476</td>
      <td>0.1580</td>
      <td>0.1340</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.1530</td>
      <td>0.1626</td>
      <td>0.1344</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.1570</td>
      <td>0.1650</td>
      <td>0.1392</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.1572</td>
      <td>0.1660</td>
      <td>0.1458</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.1604</td>
      <td>0.1686</td>
      <td>0.1552</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.1606</td>
      <td>0.1686</td>
      <td>0.1638</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.1606</td>
      <td>0.1686</td>
      <td>0.1686</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.1606</td>
      <td>0.1686</td>
      <td>0.1708</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.1606</td>
      <td>0.1686</td>
      <td>0.1726</td>
    </tr>
  </tbody>
</table>
</div>




```python
tabel_train_data_correction = {}
tabel_test_data_correction = {}
for ptype in ["Entropy","Gini_Index","Majority_Error"]:
    tabel_train_data_correction[ptype]= np.zeros(16)
    tabel_test_data_correction[ptype]= np.zeros(16)
    for i in range(16):
        trained_tree = build_tree(df_train_correction, 'y', depth = i+1, type_tree = ptype)

        Our_prediction_train_correction = df_train_correction.apply(prediction, axis=1, args = [trained_tree])
        tabel_train_data_correction[ptype][i]= 1- np.where(df_train_correction['y'] == Our_prediction_train_correction,1,0).mean()
        
        Our_prediction_test_correction = df_test_correction.apply(prediction, axis=1, args = [trained_tree])
        tabel_test_data_correction[ptype][i]= 1- np.where(df_train_correction['y'] == Our_prediction_test_correction,1,0).mean()
        print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15



```python
tabel_df_train_correction = pd.DataFrame.from_dict(tabel_train_data_correction)
tabel_df_train_correction
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entropy</th>
      <th>Gini_Index</th>
      <th>Majority_Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.1192</td>
      <td>0.1088</td>
      <td>0.1088</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.1060</td>
      <td>0.1042</td>
      <td>0.1042</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.1008</td>
      <td>0.0936</td>
      <td>0.0966</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0814</td>
      <td>0.0770</td>
      <td>0.0838</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0646</td>
      <td>0.0632</td>
      <td>0.0686</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.0510</td>
      <td>0.0522</td>
      <td>0.0618</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.0406</td>
      <td>0.0404</td>
      <td>0.0564</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.0330</td>
      <td>0.0318</td>
      <td>0.0542</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.0274</td>
      <td>0.0268</td>
      <td>0.0506</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.0218</td>
      <td>0.0214</td>
      <td>0.0480</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.0194</td>
      <td>0.0190</td>
      <td>0.0420</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.0188</td>
      <td>0.0184</td>
      <td>0.0376</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.0180</td>
      <td>0.0180</td>
      <td>0.0320</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.0180</td>
      <td>0.0180</td>
      <td>0.0256</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.0180</td>
      <td>0.0180</td>
      <td>0.0212</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.0180</td>
      <td>0.0180</td>
      <td>0.0180</td>
    </tr>
  </tbody>
</table>
</div>




```python
tabel_df_test_correction = pd.DataFrame.from_dict(tabel_test_data_correction)
tabel_df_test_correction
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entropy</th>
      <th>Gini_Index</th>
      <th>Majority_Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.1192</td>
      <td>0.1442</td>
      <td>0.1442</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.1390</td>
      <td>0.1472</td>
      <td>0.1472</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.1582</td>
      <td>0.1622</td>
      <td>0.1570</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.1680</td>
      <td>0.1676</td>
      <td>0.1618</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.1796</td>
      <td>0.1788</td>
      <td>0.1702</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.1912</td>
      <td>0.1876</td>
      <td>0.1722</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.1962</td>
      <td>0.1972</td>
      <td>0.1760</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.2086</td>
      <td>0.2062</td>
      <td>0.1778</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.2126</td>
      <td>0.2126</td>
      <td>0.1814</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.2152</td>
      <td>0.2146</td>
      <td>0.1856</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.2178</td>
      <td>0.2166</td>
      <td>0.1938</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.2196</td>
      <td>0.2182</td>
      <td>0.1980</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.2196</td>
      <td>0.2182</td>
      <td>0.2054</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.2196</td>
      <td>0.2182</td>
      <td>0.2142</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.2196</td>
      <td>0.2182</td>
      <td>0.2208</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.2196</td>
      <td>0.2182</td>
      <td>0.2224</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
