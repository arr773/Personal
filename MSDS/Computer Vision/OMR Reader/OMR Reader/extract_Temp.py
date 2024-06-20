from PIL import Image, ImageDraw
from argparse import ArgumentParser
import string
import numpy as np

image=Image.open("new.jpg")
lettersToPixels = {}
pixelsToLetters = {}
for i, letter in enumerate(string.ascii_uppercase + string.digits + ' '):
    lettersToPixels[letter] = i*7
    pixelsToLetters[i*7] = letter

pix=image.load()
arr=np.empty([100,600])
for j in range(100):
    for i in range(600):
        arr[j][i]=pix[50+i,50+j]
newarr=(np.mean(arr,axis=0))

s=""
for i in newarr:
    if i>254:
        break
    temp=i
    if temp%7!=0:
        if (temp%7)>3:
            temp=temp+(7-temp%7)
        else:
            temp=temp-temp%7
    s=s+pixelsToLetters[temp] 
out=""
for i in range(len(s)-1):
    if s[i] in string.ascii_uppercase and s[i+1]in string.digits:
        out=out+s[i]
        out=out+"\n"
    elif s[i+1] in string.ascii_uppercase and s[i]in string.digits:
        out=out+s[i]
        out=out+" "
    else:
        out=out+s[i]
out=out+s[i+1]   
# print(out)
f = open("out.txt", "w")
f.write(out)
f.close()
# print(s)






