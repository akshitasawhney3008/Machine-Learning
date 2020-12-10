import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.externals import joblib


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def prepare_data(list_of_fol):
    data = []
    for fol in list_of_fol:
        mypath = fol  # Enter Directory of all images
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        for n in range(0, len(onlyfiles)):
            image = cv2.imread(join(mypath, onlyfiles[n]))
            images = cv2.resize(image, fixed_size)
            gray_image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            gray_image = gray_image.flatten()
            data.append(gray_image)

    data_arr = np.array(data)
    label_arr = np.empty((data_arr.shape[0], 1), dtype=int)
    start = 0
    step = int(data_arr.shape[0] / 5)
    end = step
    for i in range(0, 5):
        for j in range(start, end):
            label_arr[j][0] = i
        start = start + step
        end = end + step
    return data_arr, label_arr


fixed_size = tuple((32, 32))

list_of_fol_train = ["Train_val/character_1_ka", "Train_val/character_2_kha" , "Train_val/character_3_ga", "Train_val/character_4_gha", "Train_val/character_5_kna"]
list_of_fol_test = ["Test/character_1_ka", "Test/character_2_kha" , "Test/character_3_ga", "Test/character_4_gha", "Test/character_5_kna"]


train_data_arr, train_label_arr = prepare_data(list_of_fol_train)
test_data_arr, test_label_arr = prepare_data(list_of_fol_test)

whole_data = np.append(train_data_arr, test_data_arr, axis=0)
pca = PCA(n_components=100)
pca.fit(whole_data)
whole_data = pca.transform(whole_data)

train_data_arr = whole_data[0:8500, :]
test_data_arr = whole_data[8500:, :]

train_idx_list = []
train_idx = 0
for i in train_data_arr.T:
    if np.count_nonzero(i) == 0:
        train_idx_list.append(train_idx)
    train_idx += 1


test_idx_list = []
test_idx = 0
for i in test_data_arr.T:
    if np.count_nonzero(i) == 0:
        test_idx_list.append(test_idx)
    test_idx += 1

intersection_list = intersection(train_idx_list, test_idx_list)
train_data_arr = np.delete(train_data_arr, intersection_list, axis=1)
test_data_arr = np.delete(test_data_arr, intersection_list, axis=1)

scaled_train_data = preprocessing.scale(train_data_arr)
scaled_test_data = preprocessing.scale(test_data_arr)

# C_range = np.logspace(-1, 10, 10)
# gamma_range = np.logspace(-10, 3, 10)
# param_grid = dict(gamma=gamma_range, C=C_range)

# param_grid = dict(C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
# gs = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
# gs.fit(scaled_train_data, train_label_arr.ravel())
#
# print('The parameters combination that would give best accuracy is : ')
# print(gs.best_params_)
# print('The best accuracy achieved after parameter tuning via grid search is : ', gs.best_score_)
#
# best_params = gs.best_params_
# for key, val in best_params.items():
#     if key == 'C':
#         C = best_params[key]
#     else:
#         gamma = best_params[key]
#
# sum = 0
# mysvclassifier = SVC(C=C)
#
# joblib.dump(mysvclassifier, 'mysvcclassifier.pkl')

mysvclassifier = joblib.load('mysvcclassifier.pkl')
kf = StratifiedKFold(n_splits=5)
fold = 1

list_of_train_acc = []
list_of_test_acc = []
print('For train data')
# For train data
for train, test in kf.split(scaled_train_data, train_label_arr):
    X_train = scaled_train_data[train, :]
    Y_train = train_label_arr[train, :]
    X_test = scaled_train_data[test, :]
    Y_test = train_label_arr[test, :]
    mysvclassifier = mysvclassifier.fit(X_train, Y_train.ravel())
    train_prediction = mysvclassifier.predict(X_train)
    train_acc = accuracy_score(Y_train, train_prediction)
    train_err = 1-train_acc
    test_prediction = mysvclassifier.predict(X_test)
    test_acc = accuracy_score(Y_test, test_prediction)
    test_err = 1 - test_acc
    list_of_train_acc.append(train_acc*100)
    list_of_test_acc.append(test_acc * 100)
    print('For fold number ' + str(fold))
    print('Train acc = ' + str(train_acc*100) + ' Test acc = ' + str(test_acc*100))
    print('Train err = ' + str(train_err*100) + ' Test err = ' + str(test_err*100) + '\n')
    fold += 1

print('Avg train acc = ' + str(sum(list_of_train_acc)/5))
print('Avg test acc = ' + str(sum(list_of_test_acc)/5) + '\n')

print('For test data')
# For test data
test_prediction = mysvclassifier.predict(scaled_test_data)
test_acc = accuracy_score(test_label_arr, test_prediction)
test_err = 1 - test_acc
print('Train acc = ' + str(train_acc*100) + ' Test acc = ' + str(test_acc*100))
print('Train err = ' + str(train_err*100) + ' Test err = ' + str(test_err*100) + '\n')
