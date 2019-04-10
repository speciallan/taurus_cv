import getpass
import cv2
import re

# i7-8700k
USER_LOCAL_CPU = 'speciallan'
# GTX1080Ti
USER_SERVER_GPU = 'wowkie'
# GTX1070
USER_SERVER2_GPU = 'mlg1504'

def spe(*vars):
    for var in vars:
        print(var)
    exit()

def get_base_dir():

    login_user = getpass.getuser()

    root = '/home/speciallan/Documents/python/'
    device_name = 'cpu'

    if login_user == USER_SERVER_GPU:
        root = '/home/' + USER_SERVER_GPU + '/speciallan/'
        device_name = 'gpu'

    elif login_user == USER_SERVER2_GPU:
        root = '/home/' + USER_SERVER2_GPU +'/speciallan/'
        device_name = 'gpu'

    return root, device_name

def showimg(img):
    cv2.namedWindow('page', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('page', 1000, 1000)
    cv2.imshow('page', img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

def runplot():
    plt.figure(figsize=(20,10))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

shownum = 1
def show(img, name = 'test', isgray=0, pos=0):
    global shownum
    # if isgray:
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # else:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 制定图片位置
    if pos != 0:
        plt.subplot(2, 4, pos)
    else:
        plt.subplot(2, 4, shownum)

    plt.title(name), plt.imshow(img)
    shownum += 1

    # 如果超过8个 就用最后一张图
    if shownum > 8:
        shownum -= 1
    return True

def embedded_numbers(s):
    pieces = re.compile(r'(\d+)').split(s)      # 切成数字和非数字
    pieces[1::2] = map(int, pieces[1::2])       # 将数字部分转成整数
    return pieces

def sort_string(lst):
    return sorted(lst, key=embedded_numbers)    # 将前面的函数作为key来排序

base_dir, device_name = get_base_dir()

if device_name == 'cpu':
    from matplotlib import pyplot as plt