import numpy as np
import cv2
import numpy as np
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model as tfk__load_model
import math

# IMG_SIZE =32
IMG_SIZE =45
class MathSymbol:
    def __init__(self, symbol_type, x1, y1, x2, y2):
        self.symbol_type = symbol_type
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __str__(self):
        return f"{self.symbol_type}: ({self.x1}, {self.y1}) to ({self.x2}, {self.y2})"

class MathSymbolManager:
    def __init__(self):
        self.symbols = []

    def add_symbol(self, key, symbol_type, x1, y1, x2, y2):
        self.symbols.append((key, MathSymbol(symbol_type, x1, y1, x2, y2)))

    def insert_symbol(self, index, key, symbol_type, x1, y1, x2, y2):
        if 0 <= index <= len(self.symbols):
            self.symbols.insert(index, (key, MathSymbol(symbol_type, x1, y1, x2, y2)))
        else:
            raise IndexError("Index out of range")

    def __getitem__(self, index):
        return self.symbols[index]

    def __setitem__(self, index, value):
        self.symbols[index] = value

    def __delitem__(self, index):
        del self.symbols[index]

    def __len__(self):
        return len(self.symbols)

    def modify_symbol(self, index, **kwargs):
        key, symbol = self.symbols[index]
        for attr, value in kwargs.items():
            if attr == 'key':
                self.symbols[index] = (value, symbol)
            else:
                setattr(symbol, attr, value)

    def display_all_symbols(self):
        for key, symbol in self.symbols:
            print(f"Key: {key}, Symbol: {symbol}")

def resize_with_padding(image, target_size=(IMG_SIZE, IMG_SIZE)):
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a white background of the target size
    background = np.ones((target_size[0], target_size[1]), dtype=np.uint8) * 255
    
    # Get the aspect ratio of the image
    h, w = image.shape
    aspect = w / h

    # Calculate new dimensions
    if aspect > 1:
        # Width is larger, so scale based on width
        new_w = target_size[0]
        new_h = int(new_w / aspect)
    else:
        # Height is larger or equal, so scale based on height
        new_h = target_size[1]
        new_w = int(new_h * aspect)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate position to paste the image
    x_offset = (target_size[1] - new_w) // 2
    y_offset = (target_size[0] - new_h) // 2
    
    # Paste the resized image onto the white background
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return background

def stackImages(scale,imgArray2):
    imgArray = imgArray2
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img , img2 , imgContour):
    # img is canny and img2 is normal
    image_data = {}

    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    i=1

    for cnt in contours:
        contour_mask = np.zeros(img.shape, dtype=np.uint8)
        area = cv2.contourArea(cnt)
        print(f"For i = {i} the area = {area}")
        # if area>10: 
        if area>8:   # updated to 8 on 29/07/2024
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 4)
            # cv2.drawContours(contour_mask, [cnt], -1, (255, 0, 0), 16)
            cv2.drawContours(contour_mask, [cnt], -1, (255, 0, 0), 12)   # after cuda edit

            # draw blue contours
            x, y, w, h = cv2.boundingRect(cnt)
            # print(f"xi , yi , xf , wf = {x} , {y} , {w+x} , {h+y}")

            #bounding box

            # #ploting the bounding box
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,str(i),
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)

            ###_--------------------------------------------------------------------
            #creating the mask
            # Create the background image
            background = np.ones_like(img2)*255

            # Extract the region of interest (ROI) from the original image using the mask
            roi = cv2.bitwise_and(img2, img2, mask=contour_mask)

            # Invert the mask to get the background part
            inverse_mask = cv2.bitwise_not(contour_mask)
                
            # Combine the ROI with the green background
            background = cv2.bitwise_and(background, background, mask=inverse_mask)
            result = cv2.add(background, roi)    
            ###_--------------------------------------------------------------------

            cv2.drawContours(background, cnt, -1, (0, 0, 0), 2)


            eplison_x = 3
            eplison_y = 3
            y_min = max( y-eplison_y,0 )
            y_max = min(  y+eplison_y+h   ,  img.shape[0] )
            x_min = max( x-eplison_x,0 )
            x_max = min(  x+eplison_x+w  ,  img.shape[1] )

            imgCropped = result[ y_min : y_max ,  x_min : x_max  ]    # hieght , wight 
            '''
            # imgCropped = result[y:y+h,x:x+w]    # hieght , wight  #after cuda edit
            eplison_x = 3
            eplison_y = 3
            imgCropped = result[y-eplison_y:y+eplison_y+h,x-eplison_x:x+eplison_x+w] 
            '''

            imgResize = resize_with_padding(imgCropped, target_size = ( IMG_SIZE, IMG_SIZE ) )

            # Adding a single item
            image_data[len(image_data)] = {'image': imgResize, 'xi': x, 'yi': y  , 'xf': w + x , 'yf': h + y  }


            i+=1
    return image_data

def image_to_dict_list(img):

    # img = cv2.imread(path)
    imgContour = img.copy()
    target_size=img.shape
    background = np.ones((target_size[0], target_size[1]), dtype=np.uint8) * 255
    # background = img.copy()

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh  = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    imgCanny  = thresh   # jsut so that i can use the same variable again
    # imgCanny = cv2.Canny(imgGray,50,50)  # instead of canny use threshold

    image_data= getContours(imgCanny,imgGray,imgContour)

    imgBlank = np.zeros_like(img)
    imgStack = stackImages(1,([img,imgGray],
                                [imgCanny,imgContour]))

    # uncomment to see the image
    # cv2.imshow("Stack", imgStack)
    # cv2.waitKey(0)

    # sort the dict based on xi
    sorted_image_data = dict(sorted(image_data.items(), key=lambda item: item[1]['xi']))

    #make a list of images
    imgs = [ ]
    for item in sorted_image_data.items():
            imgs.append(item[1]['image'])
            # print(item[1]['image'].shape) 
            # print(f" yi = {item[1]['yi']} ,  yf = {item[1]['yf']}  , h = {item[1]['yf']-item[1]['yi']}   , a/r = {(item[1]['yf']-item[1]['yi'])/(item[1]['xf']-item[1]['xi'])} ") 

    # add padding to the imgs
    image_padded= []

    for img in imgs:
        temp = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT,value=[255,0,0]) 
        image_padded.append(   cv2.resize(temp, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)  )

    imgs = image_padded

    return sorted_image_data , imgs    

def condition_1( manager , i ):
    return (  (i >= 0 and i<len(manager) ) and (manager[i][1].symbol_type == 'gr')  and (type( manager[i][0] ) == type(1))    )
def condition_2( manager , i ):
    return (  (i >= 0 and i<len(manager) ) and (manager[i][1].symbol_type == 'pow')  and (type( manager[i][0] ) == type(1))    )

def combine_digits(manager):
    i =0
    while i < len(manager):
        key, symbol = manager[i]
        type_sym_i = symbol.symbol_type
        
        if condition_1( manager , i ):
            if condition_1( manager , i+1 ):    # have to include the next number also
                
                x1_temp = manager[i][1].x1
                y1_temp = manager[i][1].y1

                while condition_1( manager , i + 1 ):
                    i+=1

                final=0
                times=1
                while condition_1( manager , i -1 ):
                    # print(manager[i-1][0]   , manager[i][0])
                    final = (    manager[i-1][0]    *pow(10,times)    ) +    manager[i][0]

                    times+=1

                    x2_temp = manager[i][1].x2
                    y2_temp = manager[i][1].y2

                    del manager[i]
                    del manager[i-1]

                    manager.insert_symbol(i-1, final, 'gr', x1_temp, y1_temp, x2_temp, y2_temp)
                    i=i-1
        elif condition_2( manager , i ):
            if condition_2( manager , i+1 ):    # have to include the next number also
                
                x1_temp = manager[i][1].x1
                y1_temp = manager[i][1].y1

                while condition_2( manager , i + 1 ):
                    i+=1

                final=0
                times=1
                while condition_2( manager , i -1 ):
                    # print(manager[i-1][0]   , manager[i][0])
                    final = (    manager[i-1][0]    *pow(10,times)    ) +    manager[i][0]

                    times+=1

                    x2_temp = manager[i][1].x2
                    y2_temp = manager[i][1].y2

                    del manager[i]
                    del manager[i-1]

                    manager.insert_symbol(i-1, final, 'pow', x1_temp, y1_temp, x2_temp, y2_temp)
                    i=i-1
        i+=1
    return manager

def return_math_keys(key):
    maps = { 'e':'math.e' , 'pi':'math.pi' , 'times':'*' , 'forward_slash':'/', 'cos':'math.cos' ,'sin':'math.sin' ,'tan':'math.tan','log':'math.log' }
    if key in maps:
        return maps[key]
    elif key in ['(','[','{']:
        return '('
    elif key in [')',']','}']:
        return ')'
    else:
        return str(key)

def convert_string(manager):

    equation_str = ""

    i = 0 

    # for i in range(len(manager)):
    while i  < len(manager):
        key, symbol = manager[i]
        if( symbol.symbol_type == 'gr' ):

            if(key == 'sqrt'):                                             # logic for sqrt 
                equation_str = equation_str + str("math.sqrt(")
                
                xi_sqrt , xf_sqrt = symbol.x1 ,  symbol.x2
                i+=1
                key, symbol = manager[i]
                if i  >= len(manager):
                    break

                while( ( i  < len(manager) )  and  (symbol.x1 <  xf_sqrt)    ):
                    equation_str = equation_str + return_math_keys(key)

                    i+=1
                    if ( i  < len(manager)):
                        key, symbol = manager[i]
                    else:
                        equation_str = equation_str + str(")")
                        return equation_str

                equation_str = equation_str + str(")")
                equation_str = equation_str + return_math_keys(key)
            else:
                equation_str = equation_str + return_math_keys(key)

        elif( symbol.symbol_type == 'pow'  ):
            # equation_str = equation_str + '**' +return_math_keys(key)
            # 28/07/2024 adding logic for continued eval in power
            equation_str = equation_str + '**('
            while(( i  < len(manager) ) and (symbol.symbol_type == 'pow')):
                equation_str = equation_str + return_math_keys(key)
                print("added" , return_math_keys(key))
                i+=1
                if ( i  < len(manager)):
                        key, symbol = manager[i]
            equation_str = equation_str + ')'


        i+=1
    return equation_str    

def get_symbol_type(sorted_image_data):    # gives a list of the symbol type

    y_mid =( sorted_image_data[0]['yi'] + sorted_image_data[0]['yf']  ) /2
    e1 =0
    e2 =0
    e3 =20
    sym_type=[]

    for item in sorted_image_data.items():
        # print(item[1]['xi'])
        if item[1]['yf'] > y_mid + e1:
            sym_type.append('gr')
        elif (   (item[1]['yf'] < y_mid - e2)  and   ((item[1]['yf']-item[1]['yi'])/(item[1]['xf']-item[1]['xi'])  > 0.5 )  ):   # excluding - etc
            sym_type.append('pow')
        elif   (item[1]['yf'] < y_mid - e3):
            sym_type.append('pow')
        else:
            ValueError("NONE FOUND")
    return sym_type

def predict_equation(path,imgs, target_classes):

    # model = load_model(path)

    # model.save('deploy_web_v1.h5')
    model = tfk__load_model('deploy_web_v1.h5')


    equation = []


    for img in imgs:
        img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
        # cv2.imshow("Stack", img)
        # cv2.waitKey(0)

        # img = img / 255.0

        # Convert to numpy array and add batch dimension
        img_array = np.array([img])  # This creates a (1, 32, 32, 1) shape

        # Predict
        y_prob = model.predict(img_array)

        y_pred = np.argmax(y_prob)

        equation.append(target_classes[y_pred])

    
    # for visualising the cropped images
    imgs_temp = list(imgs)
    imgStack = stackImages(1,(imgs_temp,
                            imgs_temp))

    # cv2.imshow("Cropped Images", imgStack)
    # cv2.waitKey(0)
    
    # print(equation)  # predicted output stored in a list


    return equation
def process(parent_img):
    sorted_image_data , imgs = image_to_dict_list(parent_img)

    print("the first dixt length is ")
    print(len(sorted_image_data))

    sym_type = get_symbol_type(sorted_image_data)

    print("symbol type are")
    print(sym_type)
    print("len of sym type",len(sym_type))

    sorted_image_data_temp = sorted_image_data

    # adding symbol type to dicionary 
    for i, key in enumerate(sorted_image_data.keys()):
        sorted_image_data[key]['type'] = sym_type[i]

    # print(sorted_image_data)

    target_classes = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', 
                    'cos', 'e', 'forward_slash', 'log', 'pi', 'sin', 'sqrt', 'tan', 'times']

    # equation = predict_equation('../../v2_data_aug',imgs)   
    equation = predict_equation('../../v2_data_aug',imgs,target_classes)   
    print("ammmd the eq is")
    print(equation)

    # put the data into math_symbol
    keys = equation
    keys = [int(item) if item.isdigit() else item for item in keys]
    manager = MathSymbolManager()

    xi_values = [entry['xi'] for entry in sorted_image_data.values()]
    yi_values = [entry['yi'] for entry in sorted_image_data.values()]
    xf_values = [entry['xf'] for entry in sorted_image_data.values()]
    yf_values = [entry['yf'] for entry in sorted_image_data.values()]

    for i in range(len(keys)):
        manager.add_symbol(keys[i], sym_type[i], xi_values[i], yi_values[i], xf_values[i], yf_values[i])


    # process manager
    manager_combined  = combine_digits(manager)
    final_equation = convert_string(manager_combined)


    print("final eq is " ,final_equation)
    # print("final ans is " ,eval(final_equation))
    return equation ,  final_equation , eval(final_equation)