import numpy as np

#creating an array in numpy
score=np.array([100,65,98,99,53])
print("score is :",score)

#creating an array
arr=np.array([12,34,78,98,900,26])
print("The arr is :",arr)
#creating another array
arr2=np.array([28,87,635,86,3635,3928,282737,28287,827,000000,2878774967627615136536653])
print("the array is :",arr2) 

Two Dimensional arrays
arr2d=np.array([[1232,263,273],
                [46827,28473256,2543]])
print("the two dimansional array:",arr2d)
creating second array

arr22d=np.array([[1,2,3,4],
                 [4,5,6,7]])
print("the TWo d array",arr22d)

arr222d=np.array([[2643652,376284,2346874],  // this multidimensional array is also called as matrix
                  [2653,64874,38774]])
print("the array is :",arr222d)

another array
arr3=np.array([[2344324,257,476],
               [1742176,73264,3874],  #array
               [352763,4672,8746],
               [465674,3847764,37674],
               [285764,24926,121784]])

print(arr3)


creating array by using numpy in build functions
arr=np.zeros((3,3)) #3*3 array with o values
print(arr)

another example of craeating array with zeroes
arr=np.zeros(5,5)
print(arr)

Creatinr the array with 1 the position
arr=np.ones((3,3)) #creating the matrix with 1 filled
print(arr)

craeating the array with filled by aonstant number
arr=np.full((10,10),2)
print(arr)

To creatinr the identity matrix
mat=np.eye(10)
print(mat)

craeating the matrix second identity matrix
matri=np.eye(3)
print(matri)

creating the matrix with random values
randommat=np.random.rand(3,3) # create the random values in the matrix which is in between 0 and 1
print(randommat)

Array attributes
shape of array
arr=np.array([[1,2,3,4,5],
              [6,7,8,9,10]])
print(arr.shape)

another example of size of an array
arr2=np.array([[3542,26,26],
               [2632,23,23],
               [2653,37,218],
               [23,10,73]])
print(arr2.shape) # Find out the shapeof an matrix

size of an array
arr=np.array([[25,236,27,22,276,21],
              [245,236,423,47,476,347]]) # calculate the number of elements 
print(arr.size)

determine the datatype of the elements
arr=np.array([90,65,3657])
print(arr.dtype) # gives data types of all elements

another example 
arr=np.array(["uyeert","yst","stuyt"])
print(arr.dtype)   #<u6


datermine number of dimension
arr=np.array([[[[[[[[[[[[[[[[[[2137,26472,346,
               ],[235,2776,3478,],
              [275,3478,920]]]]]]]]]]]]]]]]]]) #18 dimension
print(arr.ndim)

indexing in an array
arr=np.array([13,2461,13,23])
print(arr[3]) #return 23

another example in one dimension
arr=np.array([14423,2371,262,2872,23728])
print(arr[4])

slicing in array
arr=np.array([23,213,2,256,388,264])
print(arr[0:]) #return the array elemnts from starting index to ending index

slicing in an 2D array
arr2d=np.array([[5465,347,3647],
                [834,34,34]])
print(arr2d[:1,:2])


mathematical oprations
a=np.array([1,24,76,7])
b=np.array([4,76,98,87])
print(a+b)
print(a*b)
print(a-b)
print(a**2)
print(np.sqrt(a))

aggrigate functions
arr=np.array([2,3,4,5,6])
print(np.mean(arr))
print(np.sum(arr))
print(np.max(arr))
print(np.min(arr))
print(np.prod(arr))

reshaping and tranposeof the matrix
arr=np.array([[45,65,87,98],
              [46,87,64,74]])
print(arr)
print(arr.T)

another example of the transpose of the matrix
arr=np.array([[43,56,87,54],
              [54,76,98,65],
              [664,756,45,34],
              [64,57,98,65]])
print(arr.T)

stacking of arrays
a=np.array([1,2,3])
b=np.array([4,5,6])
print(np.vstack((a,b)))
print(np.hstack((a,b)))

filtering according the condition
arr=np.array([4,5,6,7,8,9])
print(arr[arr<=9])  #here we check the condition according to that we can filter the elments 
print(arr)

arr=np.array([[10,20,30,40],
              [20,30,40,50],
              [30,40,50,60]])
print(arr[0,2])

arr=np.array([[36,87,90,15,28],
              [35,37,35,29,45],
              [432,24,26,98,64],
              [34,29,16,28,399]])

print(arr[0,4])

print(arr[0,4])
print(arr[1,4])
print(arr[2,4])
print(arr[3,4])

filtering using the variable
arr=np.array([23,4,5,66,78,54,57,90,57])
mask=arr>5 #we can create the variable and we can store the condition in that variable and than we pass in in the array
print(arr[mask]) #basically it will return the value which is greater than 5


where function in the numpy
replace the elements with the some values according to the condition
arr=np.array([23,2,24,2,5435,355,54353,2,4])
result=np.where(arr>5,1,0)
print(result)  #[1 0 1 0 1 1 1 0 0]

with multiple condition
arr = np.array([10, 20, 30, 40, 50])
result = np.where((arr > 15) & (arr < 45),1,0)  # Elements between 15 and 45
print(result)

working with 2D arrays
arr=np.array([[43,767,98,87,5353,988],
              [46,87,98,35,76,87]])
reuslt=np.where(arr%2==0)
print(reuslt)

delete function the array
arr=np.array([23,4,545,3657,453])
result=np.delete(arr,2)
print(result)  #4 gayab

deleting the overall row 
arr=np.array([[1,2,3,4],
              [6,7,8,9],
              [23,3,46,98]])
result=np.delete(arr,1,axis=0)
print(result)


deleting the overall column
arr=np.array([[1,2,3,4],
              [6,7,8,9],
              [23,3,46,98]])
result=np.delete(arr,2,axis=1)
print(result)

deleting the multiple elemnets from the array
arr=np.array([23,4,545,3657,453])
result=np.delete(arr,[0,3]) #delete first and fourth element
print(result)

Zomato sales data (real life example) #2020 2021 2022 2023 2024 2025
sales_data = np.array([
    [101, 50000, 55000, 60000, 65000,367837,264244],  # Restaurant A
    [102, 40000, 42000, 46000, 49000, 53000, 58000],  # Restaurant B
    [103, 60000, 63000, 67000, 71000, 75000, 80000],  # Restaurant C
    [104, 30000, 35000, 40000, 45000, 50000, 55000],  # Restaurant D
    [105, 70000, 75000, 80000, 85000, 90000, 95000],  # Restaurant E
])
getting the data of A B C restaurent
print(sales_data[0:3]) # print the A B C data

print(sales_data[:3]) #  here also same work is happen print the A B C data 

print(sales_data[:,3:]) #  here we get the all rows but only column from after 3

print(sales_data[:,:]) #  here we get the all rows but only column from after 3

sales_data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
print(sales_data[0:3])#it will include the row 0 ,1,2

sales_data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
print(sales_data[2:3,1:3])# ge pahile aahet te row dakavat anni ge dudare aahet te column dakhavatat

sales_data = np.array([
    [101, 50000, 55000, 60000, 65000,367837,264244],  # Restaurant A
    [102, 40000, 42000, 46000, 49000, 53000, 58000],  # Restaurant B
    [103, 60000, 63000, 67000, 71000, 75000, 80000],  # Restaurant C
    [104, 30000, 35000, 40000, 45000, 50000, 55000],  # Restaurant D
    [105, 70000, 75000, 80000, 85000, 90000, 95000],  # Restaurant E
])
print(sales_data[:,3:]) #here the column will start from 3 and stop  at the end

sales_data = np.array([
    [101, 50000, 55000, 60000, 65000,367837,264244],  # Restaurant A
    [102, 40000, 42000, 46000, 49000, 53000, 58000],  # Restaurant B
    [103, 60000, 63000, 67000, 71000, 75000, 80000],  # Restaurant C
    [104, 30000, 35000, 40000, 45000, 50000, 55000],  # Restaurant D
    [105, 70000, 75000, 80000, 85000, 90000, 95000],  # Restaurant E
])
print(sales_data[0:,0:1]) #here the row was start from 0 to all but the column was start from the 0 and end upto 1 which means print only first column


sales_data = np.array([
    [101, 50000, 55000, 60000, 65000,367837,264244],  # Restaurant A
    [102, 40000, 42000, 46000, 49000, 53000, 58000],  # Restaurant B
    [103, 60000, 63000, 67000, 71000, 75000, 80000],  # Restaurant C
    [104, 30000, 35000, 40000, 45000, 50000, 55000],  # Restaurant D
    [105, 70000, 75000, 80000, 85000, 90000, 95000],  # Restaurant E
])
print(np.min(sales_data[0:,1:],axis=0)) # it will return the mininum values from the each of the row
print(np.max(sales_data[0:,1:],axis=0)) # it will return the maximum values from the each of the row

sales_data = np.array([
    [101, 50000, 55000, 60000, 65000,367837,264244],  # Restaurant A
    [102, 40000, 42000, 46000, 49000, 53000, 58000],  # Restaurant B
    [103, 60000, 63000, 67000, 71000, 75000, 80000],  # Restaurant C
    [104, 30000, 35000, 40000, 45000, 50000, 55000],  # Restaurant D
    [105, 70000, 75000, 80000, 85000, 90000, 95000],  # Restaurant E
])
print(np.max(sales_data[0:,1:],axis=0)) #column madhe highest element
print(np.max(sales_data[0:,1:],axis=1)) #Row madhe highest element

#average sell of the resraurent
print(np.mean(sales_data[0:,1:],axis=1))

sales_data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
print(np.cumsum(sales_data))

arr=np.array([1, 2, 3, 4,45,4,6])
print(np.cumsum(arr)) #use cumsum function

sales_data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
print(np.cumsum(sales_data[:,:],axis=0)) # along the column wise
print(np.cumsum(sales_data[:,:],axis=1)) # along the row  wise

dot product in the array and matrics
arr1=np.array([1,2,3,4,5])

arr2=np.array([6,7,8,9,10])
result=np.dot(arr1,arr2) #here it calculate the dot product
print(result)

dot product for 2D array
arr1=np.array([[1,3,4],
               [34,65,76]])
arr2=np.array([[23,34,45],
               [24,35,46],
               [53,65,87]]) #we know that there is some rules for the matrix multiplication so that we cannot do the dot product for every vector
result=np.dot(arr1,arr2)
print(result)

another oprator (@) operator for the dot product
arr1=np.array([1,2,3,4,5])

arr2=np.array([6,7,8,9,10])
result= arr1 @ arr2 # here we can calculate the dot product
print(result)

save method in the numpy
arr=np.array([1,2,3,4,5,6,7,8])
result=np.save('pratik.npy',arr)
print("save succesfully")

loading array
loaded_array=np.load('pratik.npy') #here we can read thedata from the file pratik.npy and load in the memory
print(loaded_array)

arr1=np.array([12,2,323,23,23,23]) #creating the array
result=np.save('my_arr.npy',arr1)  #saving the array
print("save successfully")

loaded_array=np.load('my_arr.npy') #loading the array
print(loaded_array)

arr1=np.array([21635,216736,32747])
arr2=np.array([35,1313,13,13,13])
np.savez('my_array.npz',first=arr1,second=arr2) #saving the multiple elements in the array
print("arrays save successully")

data=np.load('my_array.npz')
print("data load succesfully")

#acsssing tha data
print(data['first'])
print(data['second'])

#saving and loading txt data
arr=np.array([4,2,9])
np.savetxt('array.txt',arr,delimiter=',')#saving in the text format
print("array save succesfully in the text format")

#loading the array
loaded_array=np.loadtxt('array.txt',delimiter=",")
print(loaded_array) # here the format the array is the float value by default









































