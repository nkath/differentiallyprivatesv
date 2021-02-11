
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score 
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import trange
from sklearn.calibration import CalibratedClassifierCV
from calibration import CalibratedClassifierCV as cv

#-------------------------------------------------------------------------------------------------------
# parameters : 
#-------------------------------------------------------------------------------------------------------

CONCATENATED_FEATURES = True #Combined features of Rank Pooling+Max Pooling+ Average Pooling
Leave_One_Subject_Out_CV = False
One_Fold_Cross_Validation = False
Training_Testing_Validation = True
pooling_technique = 'RankPooling' # choose among: {'RankPooling' 'AveragePooling' 'MaxPooling'}


if CONCATENATED_FEATURES:
    subpath = "RP_AP_MP"
    prefix = "RP_AP_MP_"
else:
    subpath = pooling_technique
    prefix = ""

def class_proportion(data):
    labels = np.unique(data)
    for entry in data:
        labels[entry.astype(int)] += 1
    print(np.sum(labels))
    labels /= data.shape[0]    
    return labels    

def attack_model_fn(input_shape):
    """Attack model that takes target model predictions and predicts membership.
    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(Dense(128, activation="relu", input_shape=(input_shape,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

#-------------------------------------------------------------------------------------------------------
# Function svmClassify: returns classification results using linear SVM
#-------------------------------------------------------------------------------------------------------

def my_kernel(X, Y):
    return np.dot(X, Y.T)

def svmClassifyTrainTest():

    # Load both training and testing sets
##    print('-------------------------------------------------------')
##    print('Loading the', pooling_technique, ' features ...')
    print('Training: ', trainingDataPath)
##    print('Testing: ', testingDataPath)
##    print('-------------------------------------------------------')
    trainingData = np.loadtxt(trainingDataPath)
    trainingLabels = np.loadtxt(trainingLabelsPath)
    testingData = np.loadtxt(testingDataPath)
    testingLabels = np.loadtxt(testingLabelsPath)
    

    dims = np.shape(trainingData)
    print("dimensions of Training data: ", dims)

    victim_X_train, victim_X_test, victim_y_train, victim_y_test = train_test_split(
        trainingData, trainingLabels, test_size=0.7, stratify=trainingLabels
    )
    classfier = SVC(C=1.0, kernel='linear', probability = True)  # non-linear kernel is proposed by the fernando's paper

    classfier.fit(trainingData,trainingLabels)
    #victimcali = CalibratedClassifierCV(base_estimator=classfier, method='sigmoid', cv='prefit')
    #victimcali.fit(trainingData, trainingLabels)
    #print(victimcali.predict_proba(trainingData[0,:].reshape(1,-1)))
    #5/0
    #victimcali = CalibratedClassifierCV(base_estimator=classfier, method='sigmoid', cv='prefit')
    #victimcali = cv(method='sigmoid', cv='prefit', labels= classfier.classes_.astype(int))
    #victimcali.fit(victim_X_train, victim_y_train)
    #victimcali.fit(classfier.decision_function(victim_X_test), victim_y_test)
    # Evaluate the model on the testing set
    #print('   Evaluating the model')
    estimatedLabels = classfier.predict(testingData)

    accuracy = accuracy_score(testingLabels,estimatedLabels)
    weightedF1 = f1_score(testingLabels,estimatedLabels,average='weighted')
    averageF1 = f1_score(testingLabels,estimatedLabels,average='macro')
    allF1Scores = f1_score(testingLabels,estimatedLabels,average=None)
    # Print results
    print('   C = 1.0 ')
    print('   Average F1-score = %.4f' % (averageF1))
    print('   Test accuracy = %.2f %%' % (accuracy*100))
    print('   Weighted F1-score = %.4f' % (weightedF1))


    print('-------------------------------------------------------')

    #print('Script ran for %.2f seconds' % (endTime-startTime)
    #i = 5 / 0

    print('start the attack')

    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        testingData, testingLabels, test_size=0.5, stratify=testingLabels
    )
    half = int(testingData.shape[0] / 2)
    attack_X = testingData[:half,:]
    attack_Y = testingLabels[:half]
    shadowmodel = SVC(C=1.0, kernel='linear', probability = True)
    shadowmodel.fit(attack_X,attack_Y)
    #shadowcali = CalibratedClassifierCV(base_estimator=shadowmodel, method='sigmoid',cv='prefit')
    #shadowcali = cv(cv='prefit', labels= shadowmodel.classes_.astype(int))
    #shadowcali.fit(testingData, testingLabels)
    #shadowcali.fit(shadowmodel.decision_function(attacker_X_test), attacker_y_test)
    #shadowdata = np.vstack([attacker_X_train, attacker_X_test])
    shadow_prediction = shadowmodel.predict_proba(testingData)
    #shadow_prediction = shadowcali.predict_proba(shadowdata)
    #shadow_prediction = shadowcali.predict_proba(shadowmodel.decision_function(shadowdata))

    sorted_prediction = np.sort(shadow_prediction, axis=1)
    print(sorted_prediction[2,:])

    cutted_prediction = sorted_prediction[:,-3:]
    print(cutted_prediction[2,:])
    flipped_predition = np.fliplr(cutted_prediction)
    print(flipped_predition[2,:])
    
    training_ones = np.ones(attack_X.shape[0])
    training_zeros = np.zeros(testingData.shape[0] - attack_X.shape[0])
    #training_ones = np.ones(attacker_X_train.shape[0])
    #training_zeros = np.zeros(attacker_X_test.shape[0])
    in_or_not_shadow = np.concatenate((training_ones,training_zeros))
    print(in_or_not_shadow)
    attacker_model = attack_model_fn(3)
    attacker_model.fit(flipped_predition, in_or_not_shadow, epochs = 200)

    #target_predict_training = victimcali.predict_proba(classfier.decision_function(attacker_X_train))
    targetdata = np.vstack([victim_X_test, testingData])
    target_predict_training = classfier.predict_proba(trainingData)
    #target_predict_test = victimcali.predict_proba(classfier.decision_function(targetdata))
    target_predict_test = classfier.predict_proba(testingData)
    data = np.vstack([target_predict_training, target_predict_test])

    sorted_data = np.sort(data, axis=1)
    cutted_data = sorted_data[:,-3:]
    flipped_data = np.fliplr(cutted_data)

    #in_target = np.ones(trainingData.shape[0])
    #out_target = np.zeros(testingData)
    in_target = np.ones(trainingData.shape[0])
    out_target = np.zeros(testingData.shape[0])
    in_or_not_target = np.concatenate([in_target, out_target])

    results = attacker_model.predict(flipped_data)
    prediction = (results > 0.5)
    print(prediction.shape)
    prediction = prediction.flatten()
    print(prediction.shape)
    hits = np.sum(prediction == in_or_not_target)
    accuracy = hits/in_or_not_target.shape[0]
    print('Salem')
    print(accuracy)
    print("Train Victim", class_proportion(victim_y_train))
    print("Test Victim", class_proportion(victim_y_test))
    print("Train Shadow", class_proportion(attacker_y_train))
    print("Test Shadow", class_proportion(attacker_y_test))


    return averageF1*100, accuracy*100, weightedF1*100
#-------------------------------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    path = "../processed_data//phone_watch_glasses//"
    
    #### leave one subject out cross validation ######## 
    if Leave_One_Subject_Out_CV:
        
        inputDir = path+subpath+"_0_0//leave_one_subject_out//"
        subjects = np.array(["subj1","subj2", "subj3","subj4", "subj5", "subj6"])
        feature_categories = np.array(["_features","_FWDREV_features", "_hellinger_features", "_Hellinger_FWDREV_features", "_hellinger_power_features", "_hellinger_power_FWD_REV_features", "_Power_features", "_Power_FWDREV_features"])
        af1_scores = np.zeros(6)
        acc_scores = np.zeros(6)
        wf1_scores = np.zeros(6)

        if (pooling_technique == 'RankPooling'):
            num_F_CTGR = 8
        else:
            num_F_CTGR = 1

        for F_CTGR in range(num_F_CTGR):

            #print("#************************************#")
            #print("#"       ,subpath,        "#")
            #print("#* " ,feature_categories[F_CTGR] ," *#")
            #print("#*      Leva-One-Subject-Out CV     *#")
            #print("#************************************#")

            for exSubj in range(np.shape(subjects)[0]):
        
                if (CONCATENATED_FEATURES and pooling_technique == 'RankPooling'):
                    #prefix = "RP_AP_MP_"
                    trainingDataPath = inputDir+prefix+"All_except_"+subjects[exSubj]+feature_categories[F_CTGR]+".txt"
                    trainingLabelsPath = inputDir+"All_except_"+subjects[exSubj]+"_labels.txt"
                    testingDataPath = inputDir+prefix+subjects[exSubj]+feature_categories[F_CTGR]+".txt"
                    testingLabelsPath = path+pooling_technique+"_0_0//combined_with_settings//"+subjects[exSubj]+"_AllSettings_AllHands_labels.txt"
                else:
                    #Train-Test
                    trainingDataPath = path+pooling_technique+"_0_0//leave_one_subject_out//All_except_"+subjects[exSubj]+feature_categories[F_CTGR]+".txt"
                    trainingLabelsPath = path+pooling_technique+"_0_0//leave_one_subject_out//All_except_"+subjects[exSubj]+"_labels.txt"
                    testingDataPath = path+pooling_technique+"_0_0//combined_with_settings//"+subjects[exSubj]+"_AllSettings_AllHands"+feature_categories[F_CTGR]+".txt"
                    testingLabelsPath = path+pooling_technique+"_0_0//combined_with_settings//"+subjects[exSubj]+"_AllSettings_AllHands_labels.txt"
                
                af1_scores[exSubj],acc_scores[exSubj],wf1_scores[exSubj] = svmClassifyTrainTest()
                #print(subjects[exSubj])
                #print(af1_scores[exSubj],acc_scores[exSubj],wf1_scores[exSubj])

            print("#************************************#")
            print("#****** Overall Average Scores ******#")
            print("####",feature_categories[F_CTGR],"####")
            print("#************************************#")
            print("Average F1 Score : ", np.average(af1_scores))
            print("Average Accuracy : ", np.average(acc_scores))
            print("Weighted F1 Score: ", np.average(wf1_scores))
            print("#------------------------------------#")
        #### leave on subject out cross validation ######## ends
            
##################################################################################################################################################################################

    #### Train vs Test Cross Validation ########
            
    if One_Fold_Cross_Validation:
        inputDir = path+subpath+"_0_0//training_testing_sets//"
        training_file_name = "subj1_subj3_subj4_TRAINING"
        testing_file_name = "subj2_subj5_subj6_TESTING"
        feature_categories = np.array(["_features","_FWDREV_features", "_hellinger_features", "_Hellinger_FWDREV_features", "_hellinger_power_features", "_hellinger_power_FWD_REV_features", "_Power_features", "_Power_FWDREV_features"])
        
        if (pooling_technique == 'RankPooling'):
            num_F_CTGR = 8
        else:
            num_F_CTGR = 1

        for F_CTGR in range(num_F_CTGR):

            print("#************************************#")
            print("#"       ,subpath,        "#")
            print("#* " ,feature_categories[F_CTGR] ," *#")
            print("#*      One-Fold Cross-Validation   *#")
            print("#************************************#")

            print("Training on ", training_file_name, " and testing on ", testing_file_name )
            print("##################################################################################")
            trainingDataPath = inputDir+prefix+training_file_name+feature_categories[F_CTGR]+".txt"
            trainingLabelsPath = inputDir+training_file_name+"_labels.txt"
            testingDataPath = inputDir+prefix+testing_file_name+feature_categories[F_CTGR]+".txt"
            testingLabelsPath = inputDir+testing_file_name+"_labels.txt"


            svmClassifyTrainTest()
        #### leave on subject out cross validation ########

    if Training_Testing_Validation:
        inputDir = path+subpath+"_0_0//training_testing_sets//"
        training_file_name = "subj1_subj3_subj4_TRAINING"
        testing_file_name = "subj2_subj5_subj6_TESTING"

        print("#************************************#")
        print("#"       ,subpath,        "#")
        print("#*      Testing Training Validaten   *#")
        print("#************************************#")

        print("Training on ", training_file_name, " and testing on ", testing_file_name )
        print("##################################################################################")
        trainingDataPath = inputDir+prefix+training_file_name+"_hellinger_power_FWD_REV_features.txt"
        trainingLabelsPath = inputDir+training_file_name+"_labels.txt"
        testingDataPath = inputDir+prefix+testing_file_name+"_hellinger_power_FWD_REV_features.txt"
        testingLabelsPath = inputDir+testing_file_name+"_labels.txt"
        print(trainingDataPath)
        print(trainingLabelsPath)
        af1_scores[exSubj],acc_scores[exSubj],wf1_scores[exSubj] = svmClassifyTrainTest()    

