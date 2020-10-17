# import xlrd
import matplotlib.pyplot as plt
import csv
import numpy as np
# # Give the location of the file
# loc = ('data.csv', encoding='cp1252')
#
# # To open Workbook
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)
#
# # For row 0 and column 0
# print(sheet.cell_value(0, 0))
import pandas as pd
import time
import numpy as np
from numpy.random import seed

arr=[]
arr1=[]

# df=pd.read_csv('data.csv')
# ds=pd.read_csv('data.csv',usecols=['Open'],squeeze=True)
# print(ds);
def choice(i):
    if i== 1:
#        a= pd.read_csv('data.csv',delimiter=',')
         a=np.zeros(50000)
         container= pd.read_csv('data.csv',delimiter=',')
         for i in range(50000):
             a[i]=container.iloc[i][3]


         print(a)

         start = time.time()
         s=insertion_sort(a)
         end= time.time()
         tmd=end-start
         print(tmd)
         total1 = [0, tmd, 0.1, 0.2, 0.3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                   1.9, 2, 3, 4, 5, 8]
         total2 = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400,
                   3600,
                   3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200]

         print(s)
         plt.plot(total2,total1,'r')
         plt.show()
        # merge_sort(a,start)
         #print(a)
    elif i==2:
         a = pd.read_csv('data.csv')
         a = pd.read_csv('data.csv', usecols=['High'], squeeze=True)
         print(a)
    elif i==3:
         a = pd.read_csv('data.csv')
         a = pd.read_csv('data.csv', usecols=['Low'], squeeze=True)
         print(a)
    elif i ==4:
         a = pd.read_csv('data.csv')
         a = pd.read_csv('data.csv', usecols=['Close'], squeeze=True)
         arr=[]
         arr=a
         print(a)
#
# f = open('Book1.csv')
# csv_f = csv.reader(f)
# next(csv_f, None)
# attendee_emails = []
# for row in csv_f:
#   attendee_emails.append(row[1])
# for i in range(0, len(attendee_emails)):
#     attendee_emails[i] = float(attendee_emails[i])
#
# elements = list()
# times = list()

def insertion_sort(data1):
    for i in range(1,len(data1)):
        key_item = data1[i]
        pos=i
        while pos > 0 and data1[pos-1] > key_item:
            data1[pos]= data1[pos-1]
            pos -= 1
        data1[pos]= key_item
        if(key_item == 0):
            break
    return data1
    # for i in range(len(arr)):
    #     end = time.time()
    #     print(arr[i],(end-start)*1000)
    #     elements.append(attendee_emails[i])
       # elements.append(len(arr))

        # times.append((end - start)*1000)
        #
        # print(elements)
        # print(times)
        # print('*********************')

# def merge_sort(arr,start, begin=0, end=None,):
#     if end is None:
#         end = len(arr) - 1
#     if begin >= end:
#         return
#
#     mid = begin + (end - begin) // 2
#     merge_sort(arr, start, begin, mid)
#     merge_sort(arr, start, mid + 1, end)
#     return merge(arr, start, begin, end,mid)
#
# def merge(arr, start, begin, end, mid):
#     n = mid - begin + 1
#     m = end - mid
#     L = [0] * n
#     R = [0] * m
#     for i in range(0, n):
#         L[i] = arr[begin + i]
#     for j in range(0, m):
#         R[j] = arr[mid + j + 1]
#     i = 0
#     j = 0
#     for k in range(begin, end + 1):
#         if i >= n:
#             arr[k] = R[j]
#             j += 1
#         elif j >= m:
#             arr[k] = L[i]
#             i += 1
#         else:
#             if L[i] <= R[j]:
#                 arr[k] = L[i]
#                 i += 1
#             else:
#                 arr[k] = R[j]
#                 j += 1
#     end = time.time()
#     print(arr, (end - start) )
#     elements.append(attendee_emails[i])
#     times.append((end - start))

# def partition(arr, begin, end):
#     pivot = begin
#     for i in range(begin + 1,end + 1):
#         if arr[i] <= arr[begin]:
#             pivot += 1
#             arr[i], arr[pivot] = arr[pivot], arr[i]
#
#     arr[begin], arr[pivot] = arr[pivot], arr[begin]
#     return pivot
#
#
# def quicksort(arr, begin = 0, end = None):
#     if end is None:
#         end = len(arr) - 1
#     if begin >= end:
#         return
#     pivot = partition(arr, begin, end)
#     quicksort(arr, begin, pivot - 1)
#     quicksort(arr, pivot + 1, end)
#     print(arr)

print('Welcome !')
print('1.Open\n2.High\n3.Low\n4.Close')
n= int(input('Enter Your Choice'))
choice(n)

# plt.plot(range(1000))
# plt.xlim(0,25)
# yticks(np.arange(0, 1, step=0.2))
#  plt.plot(range(1000))
# plt.ylim(0, 100)
plt.xlabel('List Length')
plt.ylabel('Time Complexity')
plt.plot(arr1 , arr, label ='Heap Sort')
plt.grid()
plt.legend()
plt.show()

#plt.xticks(np.arange(0, 50000, 1000))


# import matplotlib.pyplot as plt
# import csv
#
# x = []
# y = []
#
# with open('Book1.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(int(row[0]))
#         y.append(int(row[1]))
#
# plt.plot(x,y, label='Loaded from file!')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()
# import pandas as pd
# import plotly.express as px
#
# df = pd.read_csv('Book1.csv')
#
# fig = px.line(df, x = 'AAPL_x', y = 'AAPL_y', title='Apple Share Prices over time (2014)')
# fig.show()