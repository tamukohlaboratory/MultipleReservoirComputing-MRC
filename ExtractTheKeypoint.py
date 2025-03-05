import cv2
import numpy as np
import os
import os.path
import mediapipe as mp
import sqlite3 as lite
import csv


class ExtractTheKeypoint:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic  # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    def mediapipe_detection(self, image, model):
        # print (image.shape)
        if (np.size(image) > 0):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False                   # Image is no longer writeable
            results = model.process(image)                  # Make prediction
            image.flags.writeable = True                    # Image is now writeable 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
            return image, results
        else: 
            return image, model

    def extract_keypoints(self, results_keypoint, image, option=2):
        ''' 0 all landmark (pose, face, left, right)
            1 pose, left and right without z  without normalization
            2 poser, left and right without z with normalization
        '''
        is_frame_skip = False
        
        if (option == 0):
            pose = np.array([[res.x, res.y, res.z] for res in results_keypoint.pose_landmarks.landmark]).flatten() if results_keypoint.pose_landmarks else np.zeros(33*3)
            face = np.array([[res.x, res.y, res.z] for res in results_keypoint.face_landmarks.landmark]).flatten() if results_keypoint.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results_keypoint.left_hand_landmarks.landmark]).flatten() if results_keypoint.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results_keypoint.right_hand_landmarks.landmark]).flatten() if results_keypoint.right_hand_landmarks else np.zeros(21*3)
            return np.concatenate([pose, face, lh, rh])
        elif (option == 1):
            if (results_keypoint.pose_landmarks is not None):
                pose, lh, rh = self._ext_tracking_points(results_keypoint)
                return np.concatenate([pose, lh, rh]), is_frame_skip
            else:
                flatten = []
                is_frame_skip = True
            return flatten, is_frame_skip    
        elif ((option == 2) or (option == 3)):
            height, width, _ = image.shape
            
            # check if not detected or not, to see if the person face the camera or not
            if (results_keypoint.pose_landmarks is not None):
                # nose_x = results_keypoint.pose_landmarks.landmark[0].x
                # nose_y = results_keypoint.pose_landmarks.landmark[0].y
                # print (f'{nose_x = } ; {nose_y = }')
                body, lhand, rhand = self._ext_tracking_points(results_keypoint)
                
                # Project to image plane.
                height, width, _ = image.shape
                body = self._project(body, width, height)
                lhand = self._project(lhand, width, height)
                rhand = self._project(rhand, width, height)

                base_nose_x = body[0, 0]
                base_nose_y = body[0, 1]
                leftShoulder = body[11, :]
                rightShoulder = body[12, :]

                all_keypoint = np.concatenate([body, lhand, rhand])
                # print(f'{all_keypoint[1,:]}')

                if (option==2):
                    # print(f"option2")
                    all_keypoint[:, 0] = all_keypoint[:, 0] - base_nose_x
                    all_keypoint[:, 1] = all_keypoint[:, 1] - base_nose_y
                    # print(f'{all_keypoint[1,:]}')
                    mean = all_keypoint - np.mean(all_keypoint)
                    std = mean / np.std(mean)
                    flatten = std.flatten()
                    # flatten = all_keypoint.flatten()
                elif (option==3):
                    # print(f"option3")
                    # This centers all coordinates relative to the shoulders
                    shoulder_midpoint = (leftShoulder + rightShoulder)/2
                    dScale = np.linalg.norm(leftShoulder-rightShoulder) # shoulder_width
                    ##prevent division from zero (if shoulder to close)
                    if dScale == 0:
                        dScale = 1e-6
                    # print(f'{shoulder_midpoint.shape=}')
                    all_keypoint[:, 0] = all_keypoint[:, 0] - shoulder_midpoint[0]
                    all_keypoint[:, 1] = all_keypoint[:, 1] - shoulder_midpoint[1]
                    scale = all_keypoint / dScale
                    flatten = scale.flatten()
                is_frame_skip = False

            else:
                flatten = []
                is_frame_skip = True
            return flatten, is_frame_skip

    # convert result type which is google custom array into numpy
    def _ext_tracking_points(self, result):
        body_kpts = np.zeros([33, 2])
        lhand_kpts = np.zeros([21, 2])
        rhand_kpts = np.zeros([21, 2])
        # body_upper_kpts = np.zeros([23, 2])

        if result.pose_landmarks is not None:
            _kpts = np.array([[lmark.x, lmark.y] for lmark in result.pose_landmarks.landmark])
            body_kpts[:_kpts.shape[0], :_kpts.shape[1]] = _kpts
            # take the upper body only 
            #body_upper_kpts[:, :] = body_kpts[0:23, :]
               
        if result.left_hand_landmarks is not None:
            _kpts = np.array([
                [lmark.x, lmark.y] for lmark in result.left_hand_landmarks.landmark])
            lhand_kpts[:_kpts.shape[0], :_kpts.shape[1]] = _kpts

        if result.right_hand_landmarks is not None:
            _kpts = np.array([
                [lmark.x, lmark.y] for lmark in result.right_hand_landmarks.landmark])
            rhand_kpts[:_kpts.shape[0], :_kpts.shape[1]] = _kpts
        return body_kpts, lhand_kpts, rhand_kpts

    # make the keypoint into real coordinate and make the coordinate which outside the image 
    # transform to inside the image
    def _project(self, kpts, width, height, inplace=True, option=2):
        if inplace is True:
            temp = kpts
        else:
            temp = np.copy(kpts)

        depth = max(width, height)
        xs = temp[:, 0]
        ys = temp[:, 1]
        xs = np.floor(xs * width)
        ys = np.floor(ys * height)
        xs[xs > (width - 1)] = width - 1
        ys[ys > (height - 1)] = height - 1
        temp[:, 0] = xs
        temp[:, 1] = ys
        return temp

    def process_image_extract_keypoint(self, pathFile, 
                                       actionLabel, movieNameParam, 
                                       data_path_keypoint,
                                       numberFrame=None,
                                       optionExtraction=2,
                                       is_save=False,
                                       is_flip=False):
        indexNumberFrame = 0
        resultEachImage = []
        capture = cv2.VideoCapture(pathFile)
        # Set mediapipe model 
        # print (cap.isOpened())
        if (is_flip == False):
              movieNameParam = movieNameParam
        elif (is_flip == True):
              movieNameParam = movieNameParam + "_flip" 


        allTempExtractKeypoint = [] 


        with self.mp_holistic.Holistic(min_detection_confidence=0.5, 
                                       min_tracking_confidence=0.5) as holistic:
            resultEachFrame = []

            if not capture.isOpened():
                print("Error opening video")

            while capture.isOpened():
                if (numberFrame is None):
                    pass
                else: 
                    if (numberFrame == indexNumberFrame):
                        break
              
                # check file exist or not 
                if os.path.isfile(os.path.join(pathWorkspace, 
                                               data_path_keypoint, 
                                               actionLabel, 
                                               str(movieNameParam), 
                                               str(indexNumberFrame))):
                    indexNumberFrame += 1
                    continue

                # try:
                # Read feed
                status, frame = capture.read()
                if (is_flip is True):
                    # flipHorizontal
                    frame = cv2.flip(frame, 1)

                # there is no more frame which can be process
                if status is False: 
                    # print("Video file finished. Total Frames: %d" % (capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                    # print (" ")
                    indexNumberFrame += 1
                    break
                # Make detections
                image, tempResults = self.mediapipe_detection(frame, holistic)
                
                # # Draw landmarks
                # self.draw_styled_landmarks(image, tempResults)
                # resultEachFrame.append([image,tempResults])
                
                # extract the keypoint
                keypoints, is_frame_skip = self.extract_keypoints(tempResults, 
                                                                  image, 
                                                                  option=optionExtraction)
                allTempExtractKeypoint.append(keypoints)         

                if ((is_save==True) and (is_frame_skip==False)):
                    # increment
                    indexNumberFrame += 1
                    # Break gracefully
                    new_dir = str(os.path.join(pathWorkspace, 
                                               data_path_keypoint, 
                                               actionLabel, 
                                               movieNameParam))+'/'
                    if not os.path.isdir(new_dir):
                        os.makedirs(new_dir)

                    npy_path = os.path.join(pathWorkspace,
                                            data_path_keypoint, 
                                            actionLabel, 
                                            str(movieNameParam), 
                                            str(indexNumberFrame))
                    np.save(npy_path, keypoints)
                    # cv2.imshow("show mediapipe", image)
                    # cv2.waitKey(0)
                else:
                    continue
            
            # npy_path = os.path.join(pathWorkspace,
            #                         data_path_keypoint, 
            #                         actionLabel, 
            #                         str(movieNameParam))

            # optionExtraction 2 = normalize
            # if optionExtraction == 2:
            #     tempArray = np.asarray(allTempExtractKeypoint)
            #     reshapeResult = tempArray.reshape(tempArray.shape[0], -1, 2)
            #     mean = reshapeResult - np.mean(reshapeResult)
            #     std = mean / np.std(mean)
            #     std.reshape(tempArray.shape[0], -1)
                
            #     np.save(npy_path, std)
            
            # else:
            #     np.save(npy_path, np.asarray(allTempExtractKeypoint))

            resultEachImage.append(resultEachFrame)      
            capture.release()
            # cv2.destroyAllWindows()
        return resultEachImage, indexNumberFrame
    
    def extractVideoData(self, pathWorkspace, filePathData, dataPathExtractSave, dataPathCSV, option=2):
        csvTemp = os.path.join(pathWorkspace, dataPathCSV)

        if os.path.isfile(csvTemp) and os.access(csvTemp, os.R_OK):
            print("File exists and is readable")
        else:
            print("Either the file is missing or not readable")
            fileCSV = open(os.path.join(pathWorkspace, dataPathCSV), 'w')
            # create the csv writer
            writer = csv.writer(fileCSV)
            header = ['label', 'videoName', 'TotalExtractedFrame', 'Flip']
            # write the header
            writer.writerow(header)
 
        with open(csvTemp, 'w', encoding='UTF8') as fileCSV:
            # create the csv writer
            writer = csv.writer(fileCSV)
            
            # all_lines = loadtxt(txtPath, dtype=str,comments="#", delimiter="\t", unpack=False)
            folderVideoData = os.listdir(os.path.join(filePathData))
            for subFolder in folderVideoData:
                eachFolder = os.listdir(os.path.join(filePathData, subFolder))
                for eachMovie in eachFolder:
                    pathConverToString = str(os.path.join(filePathData, subFolder, eachMovie))
                    subFolderAsLabel = str(subFolder)
                    # to remove the extension by using negative index
                    movieName = str(eachMovie)[:-4]
                    print("Video Extracted = ", movieName)
                    resultEachImage, indexNumberFrame = self.process_image_extract_keypoint( 
                                      pathFile=pathConverToString,
                                      actionLabel=subFolderAsLabel,
                                      movieNameParam=movieName,
                                      data_path_keypoint=dataPathExtractSave,
                                      numberFrame=None,
                                      optionExtraction=option,
                                      is_save=True,
                                      is_flip=False)
                    
                    # break
                    dataCSV = [subFolder, movieName, indexNumberFrame, 'False']
                    writer.writerow(dataCSV)

                    # augmented image
                    resultEachImage, indexNumberFrame = self.process_image_extract_keypoint( 
                                      pathFile=pathConverToString,
                                      actionLabel=subFolderAsLabel,
                                      movieNameParam=movieName,
                                      data_path_keypoint=dataPathExtractSave,
                                      numberFrame=None,
                                      optionExtraction=option,
                                      is_save=True,
                                      is_flip=True)
                    dataCSV = [subFolder, movieName, indexNumberFrame, 'True']
                    writer.writerow(dataCSV)
              

if __name__ == "__main__":
    # data training extract
    pathWorkspace = '/home/bra1n/Documents/signLanguage/MRC'
    variableClass = '100'
    detailFileName = 'normalization_option2'   # 'normalization_option2'  # 'without_normalization_option1'  # 'normalization_option2'
    optionParam = 2  # 1 # 2 # 3
  
    filePathTraining = os.path.join(pathWorkspace, "DataTraining{}".format(variableClass))
    filePathValidation = os.path.join(pathWorkspace, "DataValidation{}".format(variableClass))
    filePathTesting = os.path.join(pathWorkspace, "DataTesting{}".format(variableClass))
    
    DATA_PATH_TRAINING_SAVE = 'KeypointTrainingWLASL{}_{}'.format(variableClass, detailFileName)
    DATA_PATH_VALIDATION_SAVE = 'KeypointValidationWLASL{}_{}'.format(variableClass, detailFileName)
    DATA_PATH_TESTING_SAVE = 'KeypointTestingWLASL{}_{}'.format(variableClass, detailFileName)

    PATH_TRAINING_CSV = 'KeypointTrainingWLASL{}_{}.csv'.format(variableClass, detailFileName)
    PATH_VALIDATION_CSV = 'KeypointValidationWLASL{}_{}.csv'.format(variableClass, detailFileName)
    PATH_TESTING_CSV = 'KeypointTestingWLASL{}_{}.csv'.format(variableClass, detailFileName)
   
    objectExtract = ExtractTheKeypoint()
    ''' 
    0 all landmark (pose, face, left, right)
    1 pose, left and right without z without normalization
    2 pose, left and right without z with normalization
    '''

    objectExtract.extractVideoData(pathWorkspace, filePathTraining, 
                                   DATA_PATH_TRAINING_SAVE, 
                                   PATH_TRAINING_CSV, 
                                   option=optionParam)
    objectExtract.extractVideoData(pathWorkspace, filePathTesting, 
                                   DATA_PATH_TESTING_SAVE, 
                                   PATH_TESTING_CSV, 
                                   option=optionParam)
    objectExtract.extractVideoData(pathWorkspace, filePathValidation, 
                                   DATA_PATH_VALIDATION_SAVE, 
                                   PATH_VALIDATION_CSV, 
                                   option=optionParam)