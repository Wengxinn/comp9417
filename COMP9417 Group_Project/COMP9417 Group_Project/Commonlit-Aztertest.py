#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:07:14 2021

@author: MaggieGuan
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd
import zipfile

#split the training excerpt into 2834 separate text files
train = pd.read_csv("/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/train.csv", header=0)
train = train[['id','excerpt','target','standard_error']]

for i in range(len(train)):
    file = open('/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/Azertest/sample_'+str(i)+'.txt','w')
    file.write(train.loc[i,'excerpt'])
    file.close()


driver = webdriver.Chrome("/Users/MaggieGuan/Documents/WebDriver/chromedriver")
driver.get('http://161.35.202.53:8081/')

filepath = []
iter = int(2834 / 20)
for j in range(iter):
    dropdown = Select(driver.find_element_by_id("select"))
    dropdown.select_by_visible_text('English')

    for i in range(20):
        driver.find_element_by_id("infile").send_keys("/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/Aztertest/sample_"+str(20*j+i)+".txt")
    driver.find_element_by_id("submit").click()
    WebDriverWait(driver, 30)
    result = driver.find_elements_by_xpath('//*[@id="mensajeResultados"]/a')
    filepath.append([res.get_attribute('href') for res in result])
    driver.find_element_by_xpath('//*[@id="mensajeResultados"]/a').click()

dropdown = Select(driver.find_element_by_id("select"))
dropdown.select_by_visible_text('English')
for k in range(2820,2834):
    driver.find_element_by_id("infile").send_keys("/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/Aztertest/sample_"+str(k)+".txt")
driver.find_element_by_id("submit").click()
WebDriverWait(driver, 30)
result = driver.find_elements_by_xpath('//*[@id="mensajeResultados"]/a')
filepath.append([res.get_attribute('href') for res in result])
driver.find_element_by_xpath('//*[@id="mensajeResultados"]/a').click()
driver.quit()

file = [x[0] for x in filepath]
pd.DataFrame(file).to_csv("/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/AzterTestfilepath.csv",index=False)


filename = [x.split('/')[-1] for x in file]

for f in filename:
    with zipfile.ZipFile("/Users/MaggieGuan/Downloads/"+f, 'r') as zip_ref:
        zip_ref.extractall("/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/Aztertest/"+f)
        zip_ref.close()
        
#combine all results together
filedir = ["/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/Aztertest/"+x+"/results/full_results_aztertest.csv" for x in filename]
df_list = []
for fd in filedir:
    df = pd.read_csv(fd, index_col=None, header = 0)
    df_list.append(df)
    
df_overall = pd.concat(df_list, axis=0, ignore_index=True)
df_overall.to_csv("/Users/MaggieGuan/Desktop/UNI/CS/COMP9417/GroupProject/Aztertest_features.csv",index=False)        

