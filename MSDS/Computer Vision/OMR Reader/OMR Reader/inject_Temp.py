from PIL import Image, ImageDraw
from argparse import ArgumentParser
import string

f = open("./test-images/a-3_groundtruth.txt",'r')
# ans=f.read()
ans=f.read().replace(' ',"")
ans=ans.replace('\n',"")
# print(len(ans),ans)
# print(ans)

lettersToPixels = {}
pixelsToLetters = {}
for i, letter in enumerate(string.ascii_uppercase + string.digits + ' '):
    lettersToPixels[letter] = i*7
    pixelsToLetters[i*7] = letter
pixelsToLetters[255]=""
# print(lettersToPixels)
# image = Image.new("L", (600, 100), (255))
image=Image.open("./test-images/a-3.jpg")
width, height = image.size
pix = image.load()
i=0
a=[]
b=[]
for i in range(len(ans)):
    a.append(pix[50+i,50])
# print(a)
for i in range(len(ans)):
    for j in range(100):
        pix[50+i,50+j]=lettersToPixels[ans[i]]
# image.save('blank.png')
# for i in range(len(ans)):
#     b.append(pix[i,0])
# print(b)
# print(form.size)
# form.paste(image,(250,450))
image.save("new.jpg")