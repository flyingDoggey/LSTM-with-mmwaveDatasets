import numpy as np
import traceback

def read_database(dir):
    '''
    此函数用于读取文件夹内的手势数据集文件，参数为文件夹路径
    '''
    #print("Start reading data from csv file")
    dataset = []
    gesture = 0
    # 保证循环
    while True:
        # 读取路径
        path = "F:/dataSet/20230104IWR1642GestureDataset/"+dir+"/gesture_"+str(gesture+1)+".csv"
        # 向下轮换文件 一般数据集都是这种格式（方便）
        gesture = gesture + 1
        print("Open: ", path)
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=2) #skip header and null point
        except:
            print("Path not found: "+path)
            break

        FrameNumber = 1 # 帧数。因为毫米波雷达捕捉的是动态轨迹。所以一般一个手势时长是23-24帧，一帧中有1-5个点
        pointlenght = 80 # maximum number of points in array
        framelenght = 80 # maximum number of frames in arrat
        datalenght = int(len(data))
        gesturedata = np.zeros((framelenght,4,pointlenght))
        counter = 0 # 计数用，无实际含义

        # 下面要进行一段长循环，目的是根据data数据大小将所有标签对应的数据读入
        while counter < datalenght:
            velocity = np.zeros(pointlenght) # 速度
            peak_val = np.zeros(pointlenght) # 峰值
            x_pos = np.zeros(pointlenght)    # x位置
            y_pos = np.zeros(pointlenght)    # y位置
            iterator = 0

            try:
                while data[counter][0] == FrameNumber:
                    velocity[iterator] = data[counter][3]
                    peak_val[iterator] = data[counter][4]
                    x_pos[iterator] = data[counter][5]
                    y_pos[iterator] = data[counter][6]
                    iterator += 1
                    counter += 1
            except:
                print(" ")

            framedata = np.array([velocity, peak_val,x_pos,y_pos])
            try:
                gesturedata[FrameNumber - 1] = framedata
            except:
                print("Frame number out of bound", FrameNumber)
                break

            FrameNumber += 1
        # 添加入dataset
        dataset.append(gesturedata)

    print("End of the loop")
    return dataset

dir = ["close_fist_horizontally", "close_fist_perpendicularly", "hand_to_left", "hand_to_right",
                         "hand_rotation_palm_up","hand_rotation_palm_down", "arm_to_left", "arm_to_right",
                         "hand_closer", "hand_away", "hand_up", "hand_down"]

#Read gestures from choosen directory
read_database(dir[1])
