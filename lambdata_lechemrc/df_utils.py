import numpy as np
import pandas as pd

# Test functions --- they worked!! 
rand_50 = np.random.random(50)
rand_100 = np.random.random(100)


# Real functionality below
# ------------------------

# Checks for null values
def check_nulls(df):
	return df.isna().sum()

	
def train_test_val(df, target_val, feature_vals=df.columns.drop(target)):
	'''returns simple train, test, and val splits with test 
	and val equal sizes'''

	# Setting target and features (with all columns - target) for baseline
	target = target
	features = feature_vals 
	# dropping 'best_new_music' in order to not leak the data

	# setting my X and y
	X = df[features]
	y = df[target]
 
	# Train test split
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42)

	# Train val split 
	# I chose 25% so that I would make the val and test sets the same size
	X_train, X_val, y_train, y_val = train_test_split(
		X_train, y_train, test_size=0.25, random_state=42)

	return X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape
	