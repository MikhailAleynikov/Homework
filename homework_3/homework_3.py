#!/usr/bin/env python
# coding: utf-8

# # I. Numpy

# ### Импортируйте NumPy

# In[87]:


import numpy as np


# ### Создайте одномерный массив размера 10, заполненный нулями и пятым элемент равным 1. Трансформируйте в двумерный массив.

# In[88]:


array = np.zeros(10)
array[4] = 1
array = array.reshape(2,5)
print(array)


# ### Создайте одномерный массив со значениями от 10 до 49 и разверните его (первый элемент становится последним). Найдите в нем все четные элементы.

# In[116]:


array = np.random.randint(10, 49, size=39)
print(array, array[::-1], sep='\n')
even = [i for i in array if i%2==0]
print(even)


# ### Создайте двумерный массив 3x3 со значениями от 0 до 8

# In[90]:


print(np.arange(9).reshape(3, 3))


# ### Создайте массив 4x3x2 со случайными значениями. Найти его минимум и максимум.

# In[91]:


print(np.random.randint(1, 30, size=(4,3,2)))


# ### Создайте два двумерных массива размерами 6x4 и 4x3 и произведите их матричное умножение. 

# In[92]:


array_1 =  np.random.randint(1, 30, size=(6,4))
array_2 = np.random.randint(30, 40, size=(4,3))

np.matmul(array_1, array_2)


# ### Создайте случайный двумерный массив 7x7, найти у него среднее и стандартное оклонение. Нормализуйте этот массив.

# In[93]:


array = np.random.randint(1, 30, size=(7,7))

aver = array.mean()  #среднее

skv = np.nanstd(array)  #стандартное оклонение

norm = np.linalg.norm(array)  #Нормализация
norm_total = np.sum([val/norm for val in array ])

print(aver)
print(skv)
print(norm)


# # II. Pandas

# ### Импортируйте: pandas, matplotlib, seaborn

# In[94]:


import sys
#!{sys.executable} -m pip install seaborn
import matplotlib as mpl
import pandas as pd 
import seaborn as sns
import numpy as np


# ### Загрузите датасет Tips из набора датасетов seaborn

# In[95]:


tips = sns.load_dataset('tips')


# ### Посмотрите на первые 5 строчек

# In[96]:


tips.head()


# ### Узнайте сколько всего строчек и колонок в данных

# In[97]:


print(tips.shape)


# ### Проверьте есть ли пропуски в данных

# In[98]:


tips.isnull().sum()


# ### Посмотрите на распределение числовых признаков

# In[99]:


tips.describe()


# ### Найдите максимальное значение 'total_bill'

# In[100]:


tips["total_bill"].max()


# ### Найдите количество курящих людей

# In[101]:


(tips["smoker"]=="Yes").sum()


# ### Узнайте какой средний 'total_bill' в зависимости от 'day'

# In[102]:


tips.groupby("day").total_bill.mean()


# ### Отберите строчки с 'total_bill' больше медианы и узнайте какой средний 'tip' в зависимости от 'sex'

# In[103]:


tips[tips.total_bill>tips.total_bill.median()].groupby("sex").tip.mean()


# ### Преобразуйте признак 'smoker' в бинарный (0-No, 1-Yes)

# In[104]:


smoker_bin = np.where(tips["smoker"] == "Yes", 1 ,0) 


# # III. Visualization

# ### Постройте гистограмму распределения признака 'total_bill'

# In[105]:


data = tips["total_bill"]
sns.histplot(data)


# ### Постройте scatterplot, представляющий взаимосвязь между признаками 'total_bill' и 'tip'

# In[106]:


data_scatter = tips[["total_bill","tip"]]
data_scatter.plot.scatter(x="total_bill",y="tip")


# ### Постройте pairplot

# In[107]:


sns.pairplot(data_scatter)


# ### Постройте график взаимосвязи между признаками 'total_bill' и 'day'

# In[108]:


sns.catplot(x="day",y="total_bill",data=tips)


# ### Постройте две гистограммы распределения признака 'tip' в зависимости от категорий 'time'

# In[109]:


data_dinner = tips.tip[tips.time == "Dinner"]
data_lunch = tips.tip[tips.time == "Lunch"] 

data_dinner.plot(kind="hist")

data_lunch.plot(kind="hist")


# ### Постройте два графика scatterplot, представляющих взаимосвязь между признаками 'total_bill' и 'tip' один для Male, другой для Female и раскрасьте точки в зависимоти от признака 'smoker'

# In[110]:



tips["smoker"] = np.where(tips["smoker"] == "Yes", 1 ,0) 

data_scatter_yes = tips.groupby("sex")

for sex, num in data_scatter_yes:
    print(sex)
    print(num)
    num.plot.scatter(x="total_bill",y="tip",c="smoker",cmap="viridis")
data_scatter_yes    


# ## Сделайте выводы по анализу датасета и построенным графикам. По желанию можете продолжить анализ данных и также отразить это в выводах.

# In[111]:


Выводы на основе данного датасета:
    1. Средний чек у мужчин в основном не превышает медиану, так же мужчины более щедрые на чаевые
    2. У женщин средний чек более высокий, но чаевые меньше
    3. Курение ни как не влияет на средний чек или чаевые, так же курящих мужчин больше чем женщин

