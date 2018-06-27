import turicreate as tc
import os

def convertToSframe():
    image_data = tc.image_analysis.load_images('input_data', with_path=True)
    image_data['label'] = image_data['path'].apply(lambda path: os.path.basename(os.path.dirname(path)))
    image_data.save('training_data.sframe')
    image_data.explore()


def train():
    data = tc.SFrame('training_data.sframe')
    trainData, testData = data.random_split(0.75)
    model = tc.image_classifier.create(trainData, target='label')
    predictions = model.predict(testData)
    metrics = model.evaluate(testData)
    print (metrics['accuracy'])
    model.save('trained_model.model')
    model.export_coreml('MNIST.mlmodel')
