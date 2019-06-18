#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import time
import random
import pandas as pd
import numpy as np

def generate_miao_code():

    cop_code = '010'
    rec_code = '06'

    code_head = '1'
    city_code = '07'

    rand_int = str(random.randint(10, 99))

    timestamp = str(round(time.time() * 1000))
    # print(timestamp)

    code = cop_code + rec_code + code_head + city_code + rand_int + timestamp
    # print(code)

    sum = 0
    for k,v in enumerate(code):
        i = k + 1
        if i%2 == 0:
            sum += int(v) * 3
        else:
            sum += int(v) * 1
    yu = sum % 10
    check_code = (10 - yu) % 10

    # print('yu:{}'.format(check_code))

    code += str(check_code)
    # print('code:{}'.format(code))

    return code

if __name__ == '__main__':

    num = 50000
    arr = np.array([])

    f = open('code.txt', 'w')

    for i in range(num):
        # arr = np.append(arr, [generate_miao_code()], axis=-1)
        f.write(generate_miao_code() + '\n')
        time.sleep(0.001)
        print(i)

    # arr = np.delete(arr, 0)

    # print(arr)

    # pd_data = pd.DataFrame(arr)
    # pd_data.to_csv('miao_code.csv')



