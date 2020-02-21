import pandas as pd
import numpy as np
import copy



def Purity(probs, type_of_purity = "Entropy"):
    if type_of_purity == "Entropy":
        return(-1 * probs * np.log2(probs)).sum()
    elif type_of_purity == "Gini_Index":
        return(1- (probs**2).sum())
    elif type_of_purity == "Majority_Error": 
        return(1- max(probs))



def gain_info(data, attribute_column_name, label_column_name, type_of_gain = "Entropy"):
    weighted_purity = {} 
    Expected_Purity = 0
    Purity_1 =Purity(data[label_column_name].value_counts()/ len(data[label_column_name]), type_of_gain)
                                                  
    group_names = data[attribute_column_name].unique()
    grouped_data = data[[attribute_column_name, label_column_name]].groupby(attribute_column_name)
    for name in group_names:
        X = grouped_data.get_group(name)[label_column_name]
        probs = X.value_counts()/ len(X)
        weight = len(X) / len(data[label_column_name])
        a = Purity(probs, type_of_gain)
        weighted_purity[name] = [weight, a]
        Expected_Purity += weight * a 
    return(weighted_purity,  Expected_Purity, Purity_1 - Expected_Purity)



def build_tree(data, label_column_name, attributes = None, 
               values_of_attributes= None, depth= None, type_tree = "Entropy"):    
        
    if attributes == None: 
        list_of_attributes = list(data.keys()) 
        list_of_attributes.remove(label_column_name) 
        values = [list(data[key].unique()) for key in list_of_attributes] 
        values_of_attributes = dict(zip(list_of_attributes, values)) 

        
    else:
        list_of_attributes = attributes.copy()
            
    if len(list_of_attributes) == 0 or len(data[label_column_name].unique()) == 1 or depth == 0:  
        return data[label_column_name].value_counts().idxmax(axis = 0) 
    
    else:
        probs = data[label_column_name].value_counts() / len(data[label_column_name])

        
        Target_Purity = Purity(probs, type_tree) 
        
        list_of_gains = [] 
    
        for attribute in list_of_attributes:
            list_of_gains.append(gain_info(data, attribute, label_column_name, type_tree)[2])

    
        attribute_for_split = list_of_attributes[np.argmax(list_of_gains)]

    
        tree = {attribute_for_split:{}} 

        grouped_data = data.groupby(attribute_for_split)
        list_of_attributes.remove(attribute_for_split)
        
        
        values_in_attribute_for_split = values_of_attributes[attribute_for_split]
        
        
        for value in values_in_attribute_for_split: 
            if value in list(data[attribute_for_split].unique()):
                new_data = grouped_data.get_group(value).drop(columns = [attribute_for_split])
            
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



def prediction(instance, trained_tree):
    root = next(iter(trained_tree))
    if isinstance(trained_tree[root], dict):
        branch = instance[root]
        F = trained_tree[root][branch]
        if isinstance(F, dict):
            return prediction(instance, F)
        else:
            return trained_tree[root][branch]



A = open('data-desc.txt', 'r')
B = A.read()
print(B)



Columns_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
type_of_Attributes = ['numeric', 'categorical', 'categorical', 'categorical', 'binary', 'numeric', 
                      'binary', 'binary', 'categorical', 'numeric', 'categorical', 'numeric', 
                      'numeric', 'numeric', 'numeric', 'categorical', 'binary']
dic= dict(zip(Columns_names, type_of_Attributes))




df_train = pd.read_csv('bank-train.csv', names = Columns_names)
df_test = pd.read_csv('bank-test.csv', names = Columns_names)



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




trained_tree = build_tree(df_train_new, 'y', depth= None, type_tree = "Entropy")



from pprint import pprint
pprint(trained_tree) 



Our_prediction = df_train_new.apply(prediction, axis=1, args = [trained_tree])
np.where(df_train_new['y'] == Our_prediction,1,0).mean()



Our_prediction = df_test_new.apply(prediction, axis=1, args = [trained_tree])
np.where(df_test_new['y'] == Our_prediction,1,0).mean()



df_train_correction = pd.DataFrame()
df_test_correction = pd.DataFrame()
for name in list(df_train_new.keys()):
    m = df_train_new[name].value_counts().idxmax(axis = 0)
    print('Majority in column %s at trauning is ' %name, m)
    df_train_correction[name] = np.where(df_train_new[name] == 'unknown', m, df_train_new[name])
    n = df_test_new[name].value_counts().idxmax(axis = 0)
    print('Majority in column %s at test data is ' %name, n)
    df_test_correction[name] = np.where(df_test_new[name] == 'unknown', n, df_test_new[name])



df_train_correction.loc[[1806]]



df_test_correction.loc[[1806]]



df_train_correction.head()



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




tabel_df = pd.DataFrame.from_dict(tabel_train_data)
print(tabel_df)



tabel_df_test = pd.DataFrame.from_dict(tabel_test_data)
print(tabel_df_test)




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


tabel_df_train_correction = pd.DataFrame.from_dict(tabel_train_data_correction)
print(tabel_df_train_correction)



tabel_df_test_correction = pd.DataFrame.from_dict(tabel_test_data_correction)
print(tabel_df_test_correction)





