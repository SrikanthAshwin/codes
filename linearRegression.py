from numpy import *
import matplotlib.pyplot as plt
from collections import OrderedDict

#This method used for computing cost of error between the line and every point on the graph
def compute_error_for_line_given_points(b,m,points):
  #intialising the error with zero 
  totalError = 0 
  for i in range(len(points)):
    #get x, y  values
    x = points[i,0]
    y = points[i,1] 
    #get the difference of line and y value of data point, sqauring it helps to maintain positive value
    #the new point y1 can be written as mx + b which is equal to y1, as we b,m values proceeding in that way
    totalError += (y-(m*x+b))**2

  #geting the average of Cost of Error also called Cost Function
  return totalError / float(len(points))

# runner for getting towards global minima 
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
  #getting b,m
  b = starting_b
  m = starting_m
  #calculating b,m 1000 times for optimal values 
  for i in range(num_iterations):
    #update b and m with the new more accurate b and m by performing this gradient step
    b,m = step_gradient(b, m, array(points), learning_rate)
  
  #returning optimal b and m value 
  return [b,m] 

#method used to travel towards global minima  
def step_gradient(b_current, m_current, points, learningRate):
  # starting points for our gradients
  b_gradient= 0
  m_gradient= 0

  for i in range(len(points)):
    x = points[i,0]
    y = points[i,1]
    N = float(len(points))

    #computing partial derivatives, which helps to get the direction to reach global minima
    b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
    m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current)) 

  #update b and m values
  new_b = b_current - (learningRate*b_gradient)
  new_m = m_current - (learningRate*m_gradient)

  #gives b value and m value for that iteration
  return [new_b,new_m]

# method for plotting the test data and our predicted points for x which forms a line of regression
def visualize_model_predictions(Points, a0, a1):
    y_predictions = []
    x_value = []
    for i in range(0,70,1):
        y_predict = a0 + a1 * i
        x_value.append(i)
        y_predictions.append(y_predict)
    
    plt.plot(Points, 'b.', label='test')
    plt.plot(x_value, y_predictions, color='black', label='predict')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.ylabel('y_values')
    plt.xlabel('x_values')
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

def run():
  #1 collect data points
  points=genfromtxt('/content/dataset.csv',delimiter=',')
  
  points_train=points[:70]
  points_test=points[70:]
  #2 define our hyper parameters
  #how fast should our model converge?
  learning_rate=0.0001
  # y= mx+ b
  initial_b=0
  initial_m=0
  num_iterations=1000
  # print(points)
  #3 train our model
  print("Starting gradient descent at b={0}, m={1}, error={2}".format(initial_b,initial_m, compute_error_for_line_given_points(initial_b,initial_m,points_train)))
  [b, m] = gradient_descent_runner(points_train, initial_b, initial_m, learning_rate, num_iterations) #train here 
  print("After {0} iterations b = {1}, m = {2}, error = {3}".format( num_iterations, b, m, compute_error_for_line_given_points(b,m,points_train)))
  visualize_model_predictions(points_train, b, m)

if __name__ =="__main__":
  run()
