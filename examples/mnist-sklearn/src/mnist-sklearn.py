# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
import pickle
from kogu import Kogu

# Hyperparameters
gamma = 0.001  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
tol = 0.0001   # Tolerance for stopping criterion

# The digits dataset
digits = datasets.load_digits()

Kogu.load_parameters()
Kogu.update_parameters({
    "gamma": gamma,
    "tol": tol,
})

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=gamma, tol=tol)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# We save and upload trained model
model_fname = "../models/svm_model.pkl"
with open(model_fname, 'wb') as f:
    pickle.dump(classifier, f)
Kogu.upload(model_fname)

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# Create, save and upload predicted samples of the images
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
img_fname = "../reports/figures/prediction_samples.png"
plt.savefig(img_fname)
Kogu.upload(img_fname)

# Send final score of the model to Kogu
Kogu.metrics({
    "score": metrics.f1_score(expected, predicted, average="weighted"),
})
