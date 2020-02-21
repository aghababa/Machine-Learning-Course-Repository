# Decision Tree Algorithm (ID3)




```python
import pandas as pd
import numpy as np
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
    
    if attributes == None: # ?
        
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
#attributesfile = open('data-desc.txt', 'r')
```


```python
dict_of_attributes_values = {'buying': ['vhigh', 'high', 'med', 'low'], 
                      'maint':['vhigh', 'high', 'med', 'low'],
                      'doors':['2', '3', '4', '5more'],
                      'persons':['2', '4', 'more'],
                      'lug_boot':['small', 'med', 'big'],
                      'safety':['high','low', 'med']
                     }
print(dict_of_attributes_values)  
list_of_attributes = list(dict_of_attributes_values.keys())
print(list_of_attributes)
```

    {'buying': ['vhigh', 'high', 'med', 'low'], 'maint': ['vhigh', 'high', 'med', 'low'], 'doors': ['2', '3', '4', '5more'], 'persons': ['2', '4', 'more'], 'lug_boot': ['small', 'med', 'big'], 'safety': ['high', 'low', 'med']}
    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']



```python
df = pd.read_csv('train.csv', names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'] )
df1 = df
#df.head(2)
```


```python
attributes = list_of_attributes.copy()
trained_tree = build_tree(df, 'label', attributes ,dict_of_attributes_values, type_tree = "Entropy")
```


```python
from pprint import pprint
pprint(trained_tree) 
```

    {'safety': {'high': {'persons': {'2': 'unacc',
                                     '4': {'buying': {'high': {'maint': {'high': 'acc',
                                                                         'low': 'acc',
                                                                         'med': 'acc',
                                                                         'vhigh': 'unacc'}},
                                                      'low': {'maint': {'high': {'doors': {'2': 'vgood',
                                                                                           '3': {'lug_boot': {'big': 'vgood',
                                                                                                              'med': 'acc',
                                                                                                              'small': 'vgood'}},
                                                                                           '4': 'vgood',
                                                                                           '5more': 'vgood'}},
                                                                        'low': {'doors': {'2': 'good',
                                                                                          '3': 'good',
                                                                                          '4': {'lug_boot': {'big': 'good',
                                                                                                             'med': 'vgood',
                                                                                                             'small': 'good'}},
                                                                                          '5more': 'good'}},
                                                                        'med': {'lug_boot': {'big': 'vgood',
                                                                                             'med': {'doors': {'2': 'vgood',
                                                                                                               '3': 'good',
                                                                                                               '4': 'vgood',
                                                                                                               '5more': 'vgood'}},
                                                                                             'small': 'good'}},
                                                                        'vhigh': 'acc'}},
                                                      'med': {'maint': {'high': 'acc',
                                                                        'low': {'lug_boot': {'big': 'vgood',
                                                                                             'med': {'doors': {'2': 'vgood',
                                                                                                               '3': 'good',
                                                                                                               '4': 'vgood',
                                                                                                               '5more': 'vgood'}},
                                                                                             'small': 'good'}},
                                                                        'med': {'lug_boot': {'big': 'vgood',
                                                                                             'med': 'vgood',
                                                                                             'small': 'acc'}},
                                                                        'vhigh': 'acc'}},
                                                      'vhigh': {'maint': {'high': 'unacc',
                                                                          'low': 'acc',
                                                                          'med': 'acc',
                                                                          'vhigh': 'unacc'}}}},
                                     'more': {'buying': {'high': {'maint': {'high': 'acc',
                                                                            'low': 'acc',
                                                                            'med': 'acc',
                                                                            'vhigh': 'unacc'}},
                                                         'low': {'maint': {'high': {'lug_boot': {'big': 'vgood',
                                                                                                 'med': 'vgood',
                                                                                                 'small': {'doors': {'2': 'unacc',
                                                                                                                     '3': 'acc',
                                                                                                                     '4': 'acc',
                                                                                                                     '5more': 'acc'}}}},
                                                                           'low': {'lug_boot': {'big': 'vgood',
                                                                                                'med': 'vgood',
                                                                                                'small': {'doors': {'2': 'unacc',
                                                                                                                    '3': 'good',
                                                                                                                    '4': 'good',
                                                                                                                    '5more': 'good'}}}},
                                                                           'med': {'lug_boot': {'big': 'vgood',
                                                                                                'med': 'vgood',
                                                                                                'small': {'doors': {'2': 'unacc',
                                                                                                                    '3': 'good',
                                                                                                                    '4': 'good',
                                                                                                                    '5more': 'good'}}}},
                                                                           'vhigh': {'doors': {'2': {'lug_boot': {'big': 'acc',
                                                                                                                  'med': 'acc',
                                                                                                                  'small': 'unacc'}},
                                                                                               '3': 'acc',
                                                                                               '4': 'acc',
                                                                                               '5more': 'acc'}}}},
                                                         'med': {'maint': {'high': 'acc',
                                                                           'low': {'doors': {'2': {'lug_boot': {'big': 'good',
                                                                                                                'med': 'good',
                                                                                                                'small': 'unacc'}},
                                                                                             '3': 'good',
                                                                                             '4': 'good',
                                                                                             '5more': 'vgood'}},
                                                                           'med': {'doors': {'2': {'lug_boot': {'big': 'vgood',
                                                                                                                'med': 'acc',
                                                                                                                'small': 'vgood'}},
                                                                                             '3': 'vgood',
                                                                                             '4': 'vgood',
                                                                                             '5more': 'vgood'}},
                                                                           'vhigh': 'acc'}},
                                                         'vhigh': {'maint': {'high': 'unacc',
                                                                             'low': {'doors': {'2': {'lug_boot': {'big': 'acc',
                                                                                                                  'med': 'acc',
                                                                                                                  'small': 'unacc'}},
                                                                                               '3': 'acc',
                                                                                               '4': 'acc',
                                                                                               '5more': 'acc'}},
                                                                             'med': 'acc',
                                                                             'vhigh': 'unacc'}}}}}},
                'low': 'unacc',
                'med': {'persons': {'2': 'unacc',
                                    '4': {'buying': {'high': {'lug_boot': {'big': {'maint': {'high': 'acc',
                                                                                             'low': 'acc',
                                                                                             'med': 'acc',
                                                                                             'vhigh': 'unacc'}},
                                                                           'med': {'doors': {'2': 'unacc',
                                                                                             '3': 'unacc',
                                                                                             '4': {'maint': {'high': 'acc',
                                                                                                             'low': 'acc',
                                                                                                             'med': 'acc',
                                                                                                             'vhigh': 'unacc'}},
                                                                                             '5more': 'acc'}},
                                                                           'small': 'unacc'}},
                                                     'low': {'maint': {'high': 'acc',
                                                                       'low': {'doors': {'2': {'lug_boot': {'big': 'good',
                                                                                                            'med': 'acc',
                                                                                                            'small': 'acc'}},
                                                                                         '3': 'acc',
                                                                                         '4': 'good',
                                                                                         '5more': 'acc'}},
                                                                       'med': {'doors': {'2': {'lug_boot': {'big': 'good',
                                                                                                            'med': 'acc',
                                                                                                            'small': 'acc'}},
                                                                                         '3': 'acc',
                                                                                         '4': 'good',
                                                                                         '5more': 'good'}},
                                                                       'vhigh': {'lug_boot': {'big': 'acc',
                                                                                              'med': {'doors': {'2': 'unacc',
                                                                                                                '3': 'unacc',
                                                                                                                '4': 'acc',
                                                                                                                '5more': 'acc'}},
                                                                                              'small': 'unacc'}}}},
                                                     'med': {'maint': {'high': {'lug_boot': {'big': 'acc',
                                                                                             'med': 'acc',
                                                                                             'small': 'unacc'}},
                                                                       'low': {'lug_boot': {'big': 'good',
                                                                                            'med': {'doors': {'2': 'acc',
                                                                                                              '3': 'good',
                                                                                                              '4': 'good',
                                                                                                              '5more': 'good'}},
                                                                                            'small': 'acc'}},
                                                                       'med': 'acc',
                                                                       'vhigh': {'lug_boot': {'big': 'acc',
                                                                                              'med': 'unacc',
                                                                                              'small': 'unacc'}}}},
                                                     'vhigh': {'maint': {'high': 'unacc',
                                                                         'low': {'doors': {'2': 'unacc',
                                                                                           '3': 'unacc',
                                                                                           '4': 'acc',
                                                                                           '5more': 'acc'}},
                                                                         'med': {'lug_boot': {'big': 'acc',
                                                                                              'med': {'doors': {'2': 'acc',
                                                                                                                '3': 'unacc',
                                                                                                                '4': 'acc',
                                                                                                                '5more': 'acc'}},
                                                                                              'small': 'unacc'}},
                                                                         'vhigh': 'unacc'}}}},
                                    'more': {'buying': {'high': {'lug_boot': {'big': {'maint': {'high': 'acc',
                                                                                                'low': 'acc',
                                                                                                'med': 'acc',
                                                                                                'vhigh': 'unacc'}},
                                                                              'med': {'maint': {'high': 'acc',
                                                                                                'low': {'doors': {'2': 'unacc',
                                                                                                                  '3': 'acc',
                                                                                                                  '4': 'acc',
                                                                                                                  '5more': 'acc'}},
                                                                                                'med': 'acc',
                                                                                                'vhigh': 'unacc'}},
                                                                              'small': 'unacc'}},
                                                        'low': {'lug_boot': {'big': {'maint': {'high': 'acc',
                                                                                               'low': 'good',
                                                                                               'med': 'good',
                                                                                               'vhigh': 'acc'}},
                                                                             'med': {'maint': {'high': 'acc',
                                                                                               'low': {'doors': {'2': 'acc',
                                                                                                                 '3': 'good',
                                                                                                                 '4': 'good',
                                                                                                                 '5more': 'good'}},
                                                                                               'med': {'doors': {'2': 'acc',
                                                                                                                 '3': 'good',
                                                                                                                 '4': 'good',
                                                                                                                 '5more': 'good'}},
                                                                                               'vhigh': 'acc'}},
                                                                             'small': {'maint': {'high': {'doors': {'2': 'unacc',
                                                                                                                    '3': 'acc',
                                                                                                                    '4': 'acc',
                                                                                                                    '5more': 'acc'}},
                                                                                                 'low': {'doors': {'2': 'unacc',
                                                                                                                   '3': 'acc',
                                                                                                                   '4': 'acc',
                                                                                                                   '5more': 'acc'}},
                                                                                                 'med': 'acc',
                                                                                                 'vhigh': 'unacc'}}}},
                                                        'med': {'maint': {'high': {'lug_boot': {'big': 'acc',
                                                                                                'med': {'doors': {'2': 'unacc',
                                                                                                                  '3': 'acc',
                                                                                                                  '4': 'acc',
                                                                                                                  '5more': 'acc'}},
                                                                                                'small': 'unacc'}},
                                                                          'low': {'lug_boot': {'big': 'good',
                                                                                               'med': {'doors': {'2': 'acc',
                                                                                                                 '3': 'good',
                                                                                                                 '4': 'good',
                                                                                                                 '5more': 'good'}},
                                                                                               'small': 'acc'}},
                                                                          'med': 'acc',
                                                                          'vhigh': {'lug_boot': {'big': 'acc',
                                                                                                 'med': 'unacc',
                                                                                                 'small': 'unacc'}}}},
                                                        'vhigh': {'maint': {'high': 'unacc',
                                                                            'low': {'lug_boot': {'big': 'acc',
                                                                                                 'med': {'doors': {'2': 'unacc',
                                                                                                                   '3': 'acc',
                                                                                                                   '4': 'acc',
                                                                                                                   '5more': 'acc'}},
                                                                                                 'small': 'unacc'}},
                                                                            'med': {'lug_boot': {'big': 'acc',
                                                                                                 'med': {'doors': {'2': 'unacc',
                                                                                                                   '3': 'acc',
                                                                                                                   '4': 'acc',
                                                                                                                   '5more': 'acc'}},
                                                                                                 'small': 'unacc'}},
                                                                            'vhigh': 'unacc'}}}}}}}}



```python
# df['Our_prediction'] = df.apply(print, axis=1)
```


```python
#A1 = df.apply(prediction, axis=1, args = [trained_tree])
#df['Our_prediction']
```


```python
#df[['label', 'Our_prediction']]
#np.where(df['label'] == A1,1,0).mean()
```


```python
df_test = pd.read_csv('test.csv', names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'] )
#df_test.head(2)
```


```python
#B1 = df_test.apply(prediction, axis=1, args = [trained_tree])
```


```python
#np.where(condition,'value if true','value if false')
#np.where(df_test['label'] == B1,1,0).mean()
#df_test[['label', 'Our_prediction']] 
```


```python
#df_test.head(2)
```


```python
#trained_tree2 = build_tree(df1, 'label', depth = 6, type_tree = "Majority_Error" )
```


```python
#C1 = df_test.apply(prediction, axis=1, args = [trained_tree2])
```


```python
#np.where(df_test['label'] == C1,1,0).mean()
```


```python
tabel_train_data = {}
tabel_test_data = {}
for ptype in ["Entropy","Gini_Index","Majority_Error"]:
    tabel_train_data[ptype]= np.zeros(6) # [0] * 6
    tabel_test_data[ptype]= np.zeros(6)
    for i in range(6):
        trained_tree = build_tree(df, 'label', attributes ,dict_of_attributes_values, depth = i+1, type_tree = ptype)
        
        Our_prediction = df.apply(prediction, axis=1, args = [trained_tree])
        tabel_train_data[ptype][i]= 1- np.where(df['label'] == Our_prediction,1,0).mean()
        
        Our_prediction_test = df_test.apply(prediction, axis=1, args = [trained_tree])
        tabel_test_data[ptype][i]= 1- np.where(df_test['label'] == Our_prediction_test,1,0).mean()
```


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
      <td>0.302</td>
      <td>0.302</td>
      <td>0.302</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.222</td>
      <td>0.222</td>
      <td>0.301</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.181</td>
      <td>0.176</td>
      <td>0.242</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.082</td>
      <td>0.089</td>
      <td>0.130</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.027</td>
      <td>0.027</td>
      <td>0.043</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
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
      <td>0.296703</td>
      <td>0.296703</td>
      <td>0.296703</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.222527</td>
      <td>0.222527</td>
      <td>0.315934</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.196429</td>
      <td>0.184066</td>
      <td>0.262363</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.146978</td>
      <td>0.133242</td>
      <td>0.244505</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.082418</td>
      <td>0.082418</td>
      <td>0.167582</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.082418</td>
      <td>0.082418</td>
      <td>0.167582</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
