# https://youtu.be/Tm5ywTVho3U

"""
Autokeras: Automatic model search for :
    
Installation: https://autokeras.com/
    
Image classification: autokeras.ImageClassifier
Image regression: autokeras.ImageRegressor (e.g. mnist)
Text classification: autokeras.TextClassifier
Text Regression: autokeras.TextRegressor
Structured data classification: autokeras.StructuredDataClassifier  (e.g Breast cancer dataset)
Structured data regression: autokeras.StructuredDataRegressor
    
Quick test with mnist
"""

from tensorflow.keras.datasets import mnist
import autokeras as ak

(x_train, y_train), (x_test, y_test) = mnist.load_data()

clf = ak.ImageClassifier(max_trials=1)

clf.fit(x_train, y_train, epochs=1)

print("accuracy =", clf.evaluate(x_test, y_test))
