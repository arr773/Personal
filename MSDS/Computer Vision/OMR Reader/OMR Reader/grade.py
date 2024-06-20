from PIL import Image
import sys

def is_option_marked(im, x, y, radius=20):
        # Crop the image to the surrounding region
        region = crop_region(im,x,y,radius)
        # Count the occurrences of 0 in the cropped region
        zero_count = sum(1 for pixel in region.getdata() if pixel == 0)
        non_zero_count = sum(1 for pixel in region.getdata() if pixel != 0)
        total_pixels = zero_count+non_zero_count
        if zero_count>0.25*total_pixels:
            return True
        else:
            return False

def crop_region(im,x,y,radius):
    # Get the size of the image
    width, height = im.size
    
    # Calculate the coordinates for the cropping region
    left = max(0, x - radius)
    upper = max(0, y - radius)
    right = min(width, x + radius + 1)
    lower = min(height, y + radius + 1)

    # Crop the image to the surrounding region
    region = im.crop((left, upper, right, lower))
    return region

def is_manual_mark(q_num,im, x, y, radius=20):
    region = crop_region(im,x,y,radius)
    zero_count = sum(1 for pixel in region.getdata() if pixel == 0)
    if zero_count>50:
        print("zero_count_manual",zero_count,"q_num_manual", q_num)
        return True
    else:
        return False

if __name__ == '__main__':

    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 pichu_devil.py input.jpg")
    
    image_name = sys.argv[1]
    # Load an image 
    im = Image.open(f'./test-images/{image_name}')

    gray_im = im.convert("L")
    binary_img = gray_im.point(lambda p: 255 if p > 128 else 0)
    ## dict containing x and y increments and avg pixel values for option A for different columns in OMR sheet 
    optionDict = {"col1_A":(280,695),"col2_A":(715,695),"col3_A":(1150,695),"x_increment":58,"y_increment": 47}
    
    file_name = sys.argv[2]
    output = open(file_name, "w+")
    for q_num in range(85):
        marked_choices = []
        q_num=q_num+1
        if q_num==1:
            start = optionDict["col1_A"]
            x = start[0]
            y=start[1]
        elif q_num==30:
            start = optionDict["col2_A"]
            x = start[0]
            y=start[1]
        elif q_num==59:
            start=optionDict["col3_A"]
            x = start[0]
            y=start[1]
        manual_check = is_manual_mark(q_num,binary_img, x-105, y, radius=25)
        for option in ["A","B","C","D","E"]:
            is_filled = is_option_marked(binary_img, x, y, radius=20)
            if is_filled is True:
                marked_choices.append(option)
            x = x + optionDict["x_increment"]
        
        output.write(f"{q_num} {''.join(marked_choices)}")
        if manual_check is True:
            output.write(" x\n")
        else:
            output.write("\n")
        y=y+optionDict["y_increment"]  
        x=start[0]