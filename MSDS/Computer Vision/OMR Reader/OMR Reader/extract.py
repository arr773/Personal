from PIL import Image
import sys

if __name__ == '__main__':

    if(len(sys.argv) < 2):
        raise Exception("error: Please give an injected image name, final correct answers as parameters, like this: \n"
                     "python3 ./extract.py injected.jpg output.txt")
 
    increment=7
    pad = 50
    optionArea = increment*increment
    d={"A":0,"B":1*increment,"C":2*increment,"D":3*increment,"E":4*increment}
    optionMapping={0:"A",1:"B",2:"C",3:"D",4:"E"}

    injectedImage = sys.argv[1]
    groundTruth_output = sys.argv[2]

    image = Image.open(f"./test-images/{injectedImage}")
    gray_im = image.convert("L")
    binary_img = gray_im.point(lambda p: 255 if p > 128 else 0)
    binary_pix = binary_img.load()
    optionList = [0, 1, 2, 3, 4]

    groundTruth = {}
    shuffleList = [69, 18, 41, 53, 57, 30, 55, 28, 72, 3, 73, 52, 34, 46, 6, 49, 40, 36, 13, 75, 15, 48, 5, 79, 35, 64, 21, 71, 20, 19, 78, 14, 74, 47, 68, 8, 80, 60, 65, 16, 76, 4, 29, 22, 12, 39, 66, 26, 25, 7, 32, 17, 38, 54, 59, 42, 43, 27, 2, 10, 1, 31, 83, 82, 33, 58, 63, 44, 37, 50, 11, 81, 9, 23, 45, 70, 62, 51, 67, 24, 56, 61, 77, 85, 84]
    output = open(groundTruth_output, "w+")

    for question_ind in range(len(shuffleList)):
        question = shuffleList[question_ind]
        groundTruth[question] = []
        x_value = question_ind*increment
        for option in optionList:
            y_value = option*increment
            count = 0
            for horizontal_pixel in range(pad + x_value, pad + x_value + increment):
                for vertical_pixel in range(pad + y_value, pad + y_value + increment):
                    if binary_pix[horizontal_pixel,vertical_pixel]==0:
                        count +=1
                
            if count//optionArea > 0.7:
                groundTruth[question].append(optionMapping[option])
    
    final_groundTruth = {k: groundTruth[k] for k in sorted(groundTruth)}  

    for questionNum in final_groundTruth.keys():
        output.write(f"{questionNum} {''.join(final_groundTruth[questionNum])}")
        output.write('\n')
