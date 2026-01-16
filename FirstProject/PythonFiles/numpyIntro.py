import numpy as np
import matplotlib.pyplot as plt
import random
arr_2d = np.array([[1,2,3,4],[5,6,7,8]])
# To get the tyoe of the contents in the array:-
# print(arr_id.dtype)

# To get the type of the array:-
# print(type(arr_id))

# To get the dimension of the array:-
# print(arr_id.ndim)

# To get the size of the array:-
# print(arr_id.size)

# To get the shape of the array:-
# print(arr_id.shape)

# To create a ones matrix(Matrix with all elements 1)
# mx_1s = np.ones(10)
# mx_1s = np.ones((3,4))
# mx_1s = np.ones((3,4), dtype=int)
# print(mx_1s)
# print(mx_1s.dtype)

# To create a zeroes matrix (All elements zero):-
# mx_0s = np.zeros((10,5))
# print(mx_0s)

# 0 ~ "" ~ False:-
# print(bool(0))
# print(bool(""))
# print(int(False))
# print(str(False))

# To create an empty matrix:-
# mx = np.empty((5,5))
# print(mx)

# arange() method:- Syntax--> np.arange(start,end,step), end--> Not included.
# ar_1d = np.arange(1,13,2)
# print(ar_1d)

# linspace() method (Returns array containing numbers with equal difference):- Syntax--> same as arange()
# ar_1d = np.linspace(1,5,4)
# print(ar_1d)

# reshape() method (Reshapes array 1D<-->2D<-->3D):-
arr_2D = np.array([[0,1,2,3],[4,5,6,7]])
# arr_3D = arr_2D.reshape(2,2,2)
# print(arr_3D)

# ravel() method (Converts multi dimentional array to 1D):-
# print(arr_2D.ravel())

# flatten() method (Also converts multi dimentional array to 1D but it has a optional parameter making it different from ravel()):-
# print(arr_3D.flatten())

# transpose() method (Returns transpose of a given array):-
# print(arr_2D.transpose()) or print(arr_2D.T)

# Mathematical operations with two matrices:-
arr1 = np.arange(1,13).reshape(3,4);
arr2 = np.arange(1,13).reshape(4,3);
# 1.) Addition (adds respective elements):-
# print(arr1+arr2) or print(np.add(arr1,arr2))

# Subtraction (subtracts respective elements):-
# print(arr1-arr2) or print(np.subtract(arr1,arr2))

# Multiplication (multiplies respective elements):-
# print(arr1*arr2) or print(np.multiply(arr1,arr2))

# Division (divides respective elements):-
# print(arr1/arr2) or print(np.divide(arr1, arr2))

# Matrix product (actual multiplication of two matrices):-
# print(arr1.dot(arr2)) or print(arr1 @ arr2)

# print(arr1.max()) --> To find maximum value in a matrix

# print(arr1.argmax()) --> index of maximum value

# print(arr1.min())

# print(arr1.argmin()) --> index of minimum value

# print(arr1.max(0)) --> prints maximum coloumn wise.

# print(arr1.max(1)) --> prints maximum row wise.

# print(np.sum(arr1)) --> prints sum of all elements in the matrix

# print(np.sum(arr1,0)) --> prints sum coloumn wise.

# print(np.mean(arr1)) --> prints mean of all the elements of the matrix.

# print(np.sqrt(arr1)) --> prints square root of all the elements of the matrix.

# print(np.std(arr1)) --> prints standard deviation of the matrix.

# print(np.exp(arr1)) --> prints the exponent of every element int the matrix.

# print(np.log(arr1)) --> prints the natural logarithm value of every element of the matrix.

# print(np.log10(arr1)) --> prints the logarithm value base 10 of every element of the matrix.

mx = np.arange(1,101).reshape(10,10)
# print(mx[0,0]) --> prints element at 0th index of 0th index row.
# print(mx[:,0]) --> prints all elements of the coloumn at 0th index in 1D matrix.
# print(mx[:,0:1]) --> prints all elements of the coloumn at 0th index in 2D matrix.
# print(mx[1:4, 1:4]) --> 4 is excluded.
# print(mx.itemsize) --> size of every item in the matrix.

# To concatenate two matrices:-
mx1 = np.arange(1,13).reshape(3,4);
mx2 = np.arange(1,13).reshape(3,4);
# mx3 = np.concatenate((mx1,mx2)) or np.vstack((mx1,mx2,...))--> Concatenates mx1 and mx, coloumn wise(default).
# mx3 = np.concatenate((mx1,mx2),1) or np.hstack((mx1,mx2,...))--> Concatenates mx1 and mx, row wise.
mx3 = np.concatenate((mx1,mx2))
# list1 = np.split(mx3,2) --> splits the array in 2 parts coloumn wise(default)
# list1 = np.split(mx3, 2, 1) --> splits the array in 2 parts row wise
# list1 = np.split(mx3, 2, 1)
# print(list1)

# To find values of trignometric functions:-
# import matplotlib.pyplot --> To get graph of the trignometric functions.
# print(np.sin(np.pi))
# x_sin = np.arange(0,3*np.pi,0.1)
# y_sin = np.sin(x_sin)
# plt.show()

# random() method:-
# import random
# print(np.random.random((3,3)))
# print(np.random.randint(1,4,(3,3,3)))
# print(np.random.rand(3,3,3))
# np.random.seed(10) --> For fixing the following random value
# print(np.random.randn(3,3,3))
x = [1,2,3,4]
# print(np.random.choice(x)) --> returns random value from x.
# print(np.random.permutation(x)) --> returns a permutation of x.
# print(np.random.shuffle(x)) --> shuffled the elements of the original list.
# print(x)

# String operations:-
str1 = "Aditya"
str2 = "gupta"
# print(np.char.add(str1,str2)) --> Joins the two strings.
# print(np.char.lower(str1)) --> lowercases all characters in the string.
# print(np.char.upper(str1) --> uppercases all characters in the string.
# print(np.char.center(str1,60)) --> returns string with length 60 and the str1 in center and rest empty strings.
# print(np.char.center(str1,60,fillchar='*')) --> 'fillchar=' just fills up the empty string.
# print(np.char.split(str1)) --> return a list containing the string splitted by empty strings.
# print(np.char.splitlines('hello\nIndian')) --> splits the string with '\n' being a separator.
str3 = "dmy"
str4 = "dmy"
# print(np.char.join([":","/","^"], [str3,str4,str4])) --> returns a numpy array containing strings with the respective seperators(':' as the seperator of str3, '/' as the seperator of str4)
# print(np.char.replace(str3,"m","M")) --> replaces the character given as an argument in the string mentioned.
# print(np.char.equal(str3,str4)) --> returns a True/False if the strings are same or not respectively.
# print(np.char.count(str3,"d")) --> returns count of a particular character.
# print(np.char.find(str3,"d")) --> returns index of the given character in the string.
print(np.char)