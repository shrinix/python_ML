import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Delete redundant or unnecessary cols
def del_col(col,data):
    new_data = data.drop(col, axis=1)
    return new_data

# Detect and drop duplicates from the dataset.
def del_duplicates(data):
    new_data = data.drop_duplicates(keep='first')
    return new_data

#Impute missing data
def impute_col(data,filler):
    data.fillna(filler,inplace=True)
    return data

#Typecaste variables to new type
def typecast_col(col,data,types):
    new_data = data.col.astype(types)
    return new_data
  
#Replace spaces between the strings with provided chars (default is '_')
def convert_case(col,data,chars='_'):
    data = data.str.replace(' ',chars) 
    return data

#Convert all strings to LowerCase
def convert_case(col,data,chars):
    data = data.str.lower() 
    return data

# Encoding using Label Encoder or OHE to convert categorical features to numerical features
def label_encoder(data):
    le = LabelEncoder()
    data = le.fit_transform(data)
    return data

def split_file(original_file_path, train_file_path='deu-training.txt', validation_file_path='deu-validation.txt', test_file_path='deu-test.txt'):
    # Read all lines from the original file
    with open(original_file_path, 'r') as file:
        lines = file.readlines()
    
    # Shuffle the lines to ensure randomness
    #random.shuffle(lines)
    
    # Split the lines into train, validation, and test sets
    # 80% train, 10% validation, 10% test
    # do not use sklearn train_test_split
    total_lines = len(lines)
    train_size = int(0.8 * total_lines)
    validation_size = int(0.1 * total_lines)
    test_size = total_lines - train_size - validation_size
    train_lines = lines[:train_size]
    validation_lines = lines[train_size:train_size + validation_size]
    test_lines = lines[train_size + validation_size:]
    
    # Confirm that there is no data leakage
    train_validation_intersection = set(train_lines).intersection(set(validation_lines))
    assert len(train_validation_intersection) == 0, f"Data leakage detected between train and validation sets: {train_validation_intersection}"
    
    train_test_intersection = set(train_lines).intersection(set(test_lines))
    assert len(train_test_intersection) == 0, f"Data leakage detected between train and test sets: {train_test_intersection}"
    
    validation_test_intersection = set(validation_lines).intersection(set(test_lines))
    assert len(validation_test_intersection) == 0, f"Data leakage detected between validation and test sets: {validation_test_intersection}"

    # Write the split lines to their respective files
    with open(train_file_path, 'w') as file:
        file.writelines(train_lines)
    
    with open(validation_file_path, 'w') as file:
        file.writelines(validation_lines)
    
    with open(test_file_path, 'w') as file:
        file.writelines(test_lines)

#Add a main function to call the split_file function
if __name__ == "__main__":

    original_file_path = "./deu.txt"
    split_file(original_file_path)