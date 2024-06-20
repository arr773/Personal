from PIL import Image
import sys

if __name__ == '__main__':

    if(len(sys.argv) < 3):
        raise Exception("error: Please give an input image name, groundtruth answers, final output image as a parameters, like this: \n"
                     "python3 ./inject.py form.jpg answers.txt injected.jpg")
    
    image_name = sys.argv[1]
    groundtruth_name = sys.argv[2]
    output_image = sys.argv[3]
    
    f = open(f"./test-images/{groundtruth_name}",'r')

    ans = f.read()
    ans = ans.split("\n")
    x=7
    optionStart={"A":0,"B":1*x,"C":2*x,"D":3*x,"E":4*x}
    newd={}
    for i in ans:
        k=i.split(" ")
        try:
            newd[k[0]]=k[1]
        except:
            continue
    
    shuffleList = [69, 18, 41, 53, 57, 30, 55, 28, 72, 3, 73, 52, 34, 46, 6, 49, 40, 36, 13, 75, 15, 48, 5, 79, 35, 64, 21, 71, 20, 19, 78, 14, 74, 47, 68, 8, 80, 60, 65, 16, 76, 4, 29, 22, 12, 39, 66, 26, 25, 7, 32, 17, 38, 54, 59, 42, 43, 27, 2, 10, 1, 31, 83, 82, 33, 58, 63, 44, 37, 50, 11, 81, 9, 23, 45, 70, 62, 51, 67, 24, 56, 61, 77, 85, 84]
    image=Image.open(f"./test-images/{image_name}")
    pix=image.load()
    Question_start=0
    pad=50

    questionStartX=0
    increment=x
    groundTruth = newd
    for question in shuffleList:
        for answer in list(groundTruth[str(question)]):
            optionStartY=optionStart[answer]
            for horizontal_pixel in range(x): 
                for vert_pixel in range(x):
                    pix[pad+questionStartX+horizontal_pixel,pad+optionStart[answer]+vert_pixel]=0
        questionStartX=questionStartX+increment

    image.save(f"./test-images/{output_image}")
