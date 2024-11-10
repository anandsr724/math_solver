from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from dep import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve-equation', methods=['POST'])
def solve_equation():
    try:
        # Get image data
        data = request.json
        image_data = data['image'].split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        img_arr = np.array(image.convert('RGB'))
        
        # Process image
        result = process(img_arr)
        
        # Format response
        if result["success"]:
            detected_symbols = ' '.join(map(str, result['equation']))
            solution = (f"Detected symbols: {detected_symbols}\n"
                      f"Final equation: {result['final_equation']}\n"
                      f"Result: {round(result['result'], 2)}")
        else:
            solution = f"Error: {result['error_message']}"
            if result.get('equation'):
                detected_symbols = ' '.join(map(str, result['equation']))
                solution += f"\nDetected symbols: {detected_symbols}"
            if result.get('final_equation'):
                solution += f"\nFinal equation: {result['final_equation']}"
        
        return jsonify({
            'success': result["success"],
            'solution': solution
        })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'solution': f"Server error: {str(e)}"
        })


if __name__ == '__main__':
    app.run(debug=True)