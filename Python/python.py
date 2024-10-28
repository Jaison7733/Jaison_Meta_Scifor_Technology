#Hey
#1. Find sum of list elements
a=[2,4,6,8,9]
sum=0
for i in a:
    sum=sum+i
print(sum)

#2.Largest element in a list
a=[3,6,8,2,9,3]
max=0
for i in a:
  if i>max:
    max=i
print(max)

#3.Remove Duplicates in a list
a=[2,7,3,8,7,8,2,1,9,10]
b=[]
for i in a:
  for j in b:
    if i==j:
      break
  else:
    b.append(i)
print(b)

#4. Check if all elements in a list are unique

a=[4,6,3,8,2,6,6,4,8,2,4,8,6]
b=[]
for i in a:
  for j in b:
    if i==j:
      break
  else:
    b.append(i)
if len(b)==len(a):
  print("All elements are unique")
else:
  print("All elements are not unique")

#5.Program to reverse list

a=[1,8,5,9,4,3,6,5,4]
b=[]
for i in range(len(a)-1,-1,-1):
  b.append(a[i])
print(b)

#6. Count no of odd n even numbers in a list

a=[4,5,64,2,47,68,5,86,45,75,1]
even=0
odd=0
for i in a:
  if i%2==0:
    even=even+1
  else:
    odd=odd+1
print("No of even numbers: ",even)
print("No of odd numbers: ",odd)

#7.Check if a list is subset of another list

a=[2,4,5,6,3,9,8]
b=[4,5,6]
for i in b:
  for j in a:
    if i==j:
      break
  else:
    print("List is not subset")
    break
else:
  print("List is subset")

#8.Max diff btw two consecutive elements in a list
a=[2,3,4,7,4,9]
max=0
for i in range(len(a)-1):
  diff=a[i+1]-a[i]
  if diff>max:
    max=diff
print(max)

#9.Merge Multiple dictionaries
a={"a":1,"b":2}
b={"c":'x',"d":'y'}
a.update(b)
print(a)

#10.Find words frequency in a sentence
text="The value of value is value"
a=text.split()
b={}
for i in a:
  if i in b:
    b[i]=b[i]+1
  else:
    b[i]=1
print(b)

