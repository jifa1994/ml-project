import numpy as numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn import neighbors
import csv
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

t1=time.time()

def load_data(file_name,train_data,target_data,test_data,test_target,split_percentage):
	with open(file_name,'r') as csvfile:
		line = csv.reader(csvfile)
		dataset = list(line)
		for i in range(1,len(dataset)):
			temp=[]
			dataset[i][8]=float(dataset[i][8])
			target_data.append(dataset[i][8])
			for j in range(8):
				dataset[i][j]=float(dataset[i][j])
				temp.append(dataset[i][j])
			train_data.append(temp)
	x_train, x_test, y_train, y_test = train_test_split(train_data,target_data,test_size=split_percentage,random_state=0)
	train_data = x_train
	target_data = y_train
	test_data = x_test
	test_target = y_test
	# print("test_target",test_target)
	return train_data,target_data,test_data,test_target

def res(res):
	with open(r'res_kaggle_pima.csv','w',newline='') as result:
		writer = csv.writer(result)
		count=0
		for i in res:
			count+=1
			temp=[count]
			temp.append(i)
			writer.writerow(temp)

def main():
	train_data=[]
	target_data=[]
	test_data=[]
	test_target=[]
	split_percentage = input("Input the split percentage of test data among original datasets:")
	split_percentage = float(split_percentage)
	print("Loading data.......")
	train_data,target_data,test_data,test_target=load_data(r'diabetes.csv',train_data,target_data,test_data,test_target,split_percentage)
	myList = list(range(1,100))
	neighbors1 = filter(lambda x:x,myList)
	cv_scores = []
	for k in neighbors1:
		knn = neighbors.KNeighborsClassifier(n_neighbors = k)
		scores = cross_val_score(knn, train_data, target_data, cv = 10, scoring = 'accuracy')
		cv_scores.append(scores.mean())
	mse = [1-x for x in cv_scores]
	optimal_k=mse.index(min(mse))
	print("the optimal number of neighbors is:"+ str(optimal_k+1))

	knn = neighbors.KNeighborsClassifier(n_neighbors=optimal_k+1)
	print("Starting training KNN model.......")
	knn.fit(train_data,target_data)
	print("Training complete!! Starting classifying.......")
	prediction = knn.predict(test_data)
	res(prediction)

	count=0
	for i in range(len(prediction)):
		if prediction[i]==test_target[i]:
			count+=1
	accuracy = count/len(test_target)
	print("Accuracy is", accuracy)
	diad=pd.read_csv('diabetes.csv')
	sns.countplot(x='Age',data=diad)
	plt.show()


if __name__ == '__main__':
	main()

t2=time.time()
a=t2-t1
minute=a//60
hour=minute//60
second=a-hour*3600-minute*60
print("Time using totally: "+str(hour)+"h "+str(minute)+"min "+str(second)+"s ")


