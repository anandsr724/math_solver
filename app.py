import base64
from io import BytesIO
from PIL import Image
# import your_ml_model  # Import your ML model here
from dep import *
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
# app = Flask(__name__, static_url_path='', static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve-equation', methods=['POST'])
def solve_equation():
    data = request.json
    image_data = data['image'].split(',')[1]  # Remove the "data:image/png;base64," part
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Save the image temporarily if needed
    # image.save('temp_equation.png')
    img_arr = np.array(image.convert('RGB'))
    
    # Use your ML model to solve the equation
    # solution = your_ml_model.solve(image)
    # equation ,  final_equation , final_ans = process('temp_equation.png')
    equation ,  final_equation , final_ans = process(img_arr)
    solution = final_equation +" =  " +str(round(final_ans,2))
    
    return jsonify({'solution': solution})

if __name__ == '__main__':
    app.run(debug=True)