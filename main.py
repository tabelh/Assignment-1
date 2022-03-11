#Import libraries
import numpy as np
import math
from mip import Model, xsum, minimize, BINARY, INTEGER
import pandas as pd
from itertools import product

#data for each truck
data = {
  "rent": [295,325,395], #daily rent price per truck
  "capacity_truck": [13,16,20], #capacity per truck
  "travel_cost": [0.25,0.2,0.3], #cost travelled unit distance (€)
  "emissions": [0.012,0.015,0.017], #CO2 Emissions per unit of distance
  "route_length": [500,600,700], #maximum route length per truck
  "distance_to_time": [0.1,0.2,0.15], #unit of distance to time
  "number_of_vehicles": [3,3,2], #maximum number of vehicles of this type
  "servicetime_per_customer": [10,9,9] #service time per customer
}

#coordinates of the depots
depots = {
  "x": [25,1], #x coordinates
  "y": [5,50]  #y coordinates
}

#create the dataframes
df_data = pd.DataFrame(data, index=["ModelA","ModelB","ModelC"])
df_data = df_data.T
df_depots = pd.DataFrame(depots, index=["Depot1","Depot2"])
df_depots = df_depots.T

#read a scenario according to the given file
def readScenario(FileName):
  #Define Global variables, first use in this function
  global numCustomers
  global numDepots
  global customerLocation
  global customerDemand
  
  with open(FileName, 'r') as Scenario:
      lines = Scenario.readlines()
      numCustomers = int(lines[0]) #import the amount of customers
      numDepots = int(lines[1]) #import the amount of depots
      customerLocation = np.zeros((numCustomers, 2), dtype='f') #import the customers locations
      customerDemand = np.zeros((numCustomers, 1), dtype='f') #import the demand per customer
      #split the values and make them float for each customer 
      for customers in range(2, numCustomers + 2):
        curLine = str(lines[customers])
        x = curLine.split()
        customerLocation[customers-2][0] = float(x[0]) #float the values
        customerLocation[customers-2][1] = float(x[1]) #float the values
      #link the demand to the customer in sequence  
      for reqProducts in range(numCustomers + 2, numCustomers*2 + 2):
          customerDemand[reqProducts- (numCustomers + 2)] = float(lines[reqProducts])

#Formula for euclidean Distance
def euclideanDistance(customer1,customer2):
  distance = math.sqrt(math.pow(abs(customer1[0]- customer2[0]),2)+math.pow(abs(customer1[1]-customer2[1]),2))
  #print(f'distance between {customer1} and {customer2} is {distance}')
  return distance

#Build a distance matrix to 
def createDistanceMatrix():
  global DistanceMatrix
  DistanceMatrix = np.zeros((numCustomers, numCustomers)) #initialize distance matrix
  for i in range(numCustomers): #loop over all customers 
    for j in range(0, i):
      if i == j:
        DistanceMatrix[i][j] = 0 #if start and end at same  
      else:
        DistanceMatrix[i][j] = euclideanDistance(customerLocation[i], customerLocation[j]) #Calculate distance between customer 1 and 2
        DistanceMatrix[j][i] = DistanceMatrix[i][j] #distance the otherway is the same

def main():
  #read scenario file
  readScenario('Scenario1.txt')
  #build distance matrix
  createDistanceMatrix()
  #Instantiate model
  model = Model()
  TruckCapacity = []
  
  for k in range(0,8):
      if k<3: 
          i = 0
      elif k < 6: 
          i = 1
      else:
          i = 2
      TruckCapacity.append(df_data.iloc[2][i])
      
          
      

  # number of nodes and list of vertices
  n, V, Vd = (numCustomers), set(range(len(DistanceMatrix))), set(range(len(df_depots)))
  # binary variables indicating if arc (i,j) is used on the route for vehicle K
  x = [[[model.add_var(var_type=BINARY) for k in range(0,8)] for j in V] for i in V] 
  print(len(x))
  print(len(x[0]))
  print(len(x[0][0]))
  #Binary variable indicating which is a certain bus is used
  bt = [model.add_var(var_type=BINARY) for k in range(0,8)]
  # continuous variable to prevent subtours: each city will have a different sequential id in the planned route except the first one
  y = [model.add_var() for i in V]
  
  #objective
  model.objective = minimize(xsum(x[i][j][k]*DistanceMatrix[i][j] for k in range(0,8) for i in V for j in V ))

  #constraints
  #leave each city only once
  for i in V:
    model += xsum(x[i][j] for j in V-{i}) == 1 
  #enter each city only once
  for i in V:
    model += xsum(x[j][i] for j in V-{i}) == 1 
  #subtour elimination
  for(i,j) in product(V, V):
    if i!=j : 
      model += y[i] - (n+1)*x[i][j] >= y[j]-n
  #items taken on subtour cannot exceed the trucks capacity
  for k in range(0,8):  
      for(i,j) in product(V, V):
          model += xsum(x[i][j][k]*customerDemand[j]*bt[k]) <= TruckCapacity[k]
  #no more than 1 truck can be used per truck
  for k in range(0,8):
    model += bt[k] <= 1
    
  #Optimize model
  model.optimize()

  #checking if a solution
                
main()