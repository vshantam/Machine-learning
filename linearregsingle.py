#importing library
import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import pandas  as pd


# class definition
class linear(object):

	#init method
	def __init__(self,theta_1=None,theta_2=None):
		self.theta_1=0;
		self.theta_2=0;
	
	#defining cost funtion
	@classmethod
	def CostFunction_h(self,theta_1,theta_2,x):
		return theta_1 + ( theta_2 * x )
	
	#defining partial derivation for y=mx+c type functions.here m,c is nothing but theta_1,theta_2
	@classmethod
	def Delta(self,theta_1,theta_2,x,y):
		
		#initializing sum variable for easy implementation
		total_sum_1=0
		total_sum_2=0

		#iterating over the function call		
		for i in range(len(y)):
			total_sum_1 += (theta_1 + theta_2 * x[i]) - y[i];
			total_sum_2 += ((theta_1 + theta_2 * x[i]) - y[i]) * x[i];
		total=[total_sum_1,total_sum_2];
		
		#returning the final set of values
		return total

	#defining alpha value here alpha is learning rate
	@classmethod
	def alpha(self):
		#returning the fixed value.can be changed based on the data set optimisation
		return 0.3 

	#No of test Cases
	@classmethod
	def m(self):
		return len(y)

	#rescaling the feature from -1 to +1
	@classmethod
	def rescale(self,x,y):
		for i in range(len(x)):
			x[i] = (x[i] - (sum(x)/len(x)))/(max(x)-min(x));
			y[i] = (y[i] - (sum(y)/len(y)))/(max(y)-min(y));
		return x , y

	#implementing gradient descent algorithm
	@classmethod
	def gradient(self,alpha,m,theta_1,theta_2,x,y):
		
		#setting alpha and no. of training set length
		alp=self.alpha
		m1=self.m
		
		#iteration over the optimising functions
		for i in range(int(input("No of Epochs:"))):
			print("epoch---->{}".format(i+1))
			time.sleep(0.5)
			delt=self.Delta(theta_1,theta_2,x,y)
			temp1 = theta_1 -  0.3/len(y) * (delt[0]);			
			temp2 = theta_2 -  0.3/len(y) * (delt[1]);

			#simultaneous updation of theta function		
			theta_1 = temp1;
			theta_2 = temp2;
			print(theta_1,theta_2)

		#returning the optimised set of theta
		return [theta_1,theta_2]
	
	#graph plotting function
	@classmethod
	def visualise(self,x1,y1,theta_1,theta_2):
		time.sleep(3)
		#scatter plotting the training data
		plt.scatter(x1,y1)
		#labeling the graph
		plt.xlabel("size")
		plt.ylabel("cost")
		plt.title("Cost Prediction")
		#implemeting the legend funtion
		red_patch = mpatches.Patch(color='blue', label='The Data')
		plt.legend(handles=[red_patch])
		#displaying the graph
		plt.show()
	
	#dataset loading method
	@classmethod
	def load_data(self,path):
		df=pd.read_csv(path)
		#this is linear regression with one variable hence it has only one variable i.e x
		x=list(df.iloc[:,0])
		#and onlu one output feature i.e y
		y=list(df.iloc[:,1])
		return x,y

#calling main function
if __name__=='__main__':
	#creating an class object
	obj=linear()
	#calling classmethod to load data from csv file		
	x,y=[10,11,12,13,14,15,16,17],[1,2,3,4,5,6,7,8] # added example for understanding purposes .you can load the data.
	#rescaling the features.
	x,y=obj.rescale(x,y)
	#applying gradiend descent algorithm to optimise the algorithm
	res=obj.gradient(obj.alpha,obj.m,0,0,x,y)
	print('Theta_1 is {}\ntheta_2 is {}\n'.format(res[0],res[1]))
	print(obj.CostFunction_h(res[0],res[1],int(input("Please enter the size to predict cost :"))))
	print("Do you want to visualize the data:")
	ans=input().strip()
	if ans.lower()[0]=='y':
		obj.visualise(x,y,res[0],res[1])

