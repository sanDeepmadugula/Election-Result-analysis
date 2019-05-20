#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pylab import *


# In[2]:


import os
os.chdir('C:\\Analytics\\MachineLearning\\Election result analysis')


# In[3]:


LS09Cand = pd.read_csv('LS2009Candidate.csv')
print(LS09Cand.shape)
LS09Cand.head()


# In[4]:


LS14Cand = pd.read_csv('LS2014Candidate.csv')
print(LS14Cand.shape)
LS14Cand.head()


# In[5]:


# Merging both 2009 and 2014
LS0914 = pd.concat([LS09Cand,LS14Cand])
print(LS0914.shape)
LS0914.head()


# In[7]:


# checking different parties abbrevation
LS0914['Party Abbreviation'].unique()


# In[12]:


LS0914['Party Abbreviation'].value_counts()


# In[11]:


LS0914['Party Abbreviation'].isna().sum()


# In[14]:


LS0914.columns


# In[15]:


# Introducing Alliance column for optimized substitution of Winning Party Abbrevation column

LS0914['Alliance'] = LS0914['Party Abbreviation']
LS0914['Alliance'] = LS0914['Alliance'].replace(to_replace=['INC','NCP','RJD','DMK',
                                                           'IUML', 'JMM','JD(s)','KC(M)','RLD',
                                                            'RSP','CMP(J)','KC(J)','PPI','MD'],
                                                value='UPA')

LS0914['Alliance'] = LS0914['Alliance'].replace(to_replace=['BJP','SS', 'LJP', 'SAD', 
                                                            'RLSP', 'AD','PMK','NPP','AINRC',
                                                            'NPF','RPI(A)','BPF','JD(U)','SDF',
                                                            'NDPP','MNF','RIDALOS','KMDK','IJK',
                                                            'PNK','JSP','GJM','MGP','GFP','GVP',
                                                            'AJSU','IPFT','MPP','KPP','JKPC','KC(T)'
                                                            ,'BDJS','AGP','JSS','PPA','UDP','HSPDP',
                                                            'PSP','JRS','KVC','PNP','SBSP','KC(N)','PDF','MDPF'],value='NDA')


LS0914['Alliance'] = LS0914['Alliance'].replace(to_replace=['YSRCP','AAAP', 'IND', 'AIUDF', 'BLSP', 'JKPDP', 
                                                            'JD(S)', 'INLD', 'CPI', 'AIMIM', 'KEC(M)','SWP',
                                                            'NPEP', 'JKN', 'AIFB', 'MUL', 'AUDF', 'BOPF', 
                                                            'BVA', 'HJCBL', 'JVM','MDMK'],value='Others')

LS0914


# In[18]:


# Winning seats distribution by Major Political Parties & Alliances for 2009 and 2014

SeatsWin = LS0914[(LS0914.Position==1)].groupby(['Alliance','Year'])['Position'].sum().reset_index().pivot(index='Alliance', columns='Year',values='Position').reset_index().fillna(0).sort_values([2014,2009], ascending=False).reset_index(drop = True)
        
print(SeatsWin['Alliance'].unique())
SeatsWin


# In[19]:


# Plotting pie chart
plt.pie(SeatsWin[2009])
# adding circle at center
my_circle = plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() # to get current figure
ax = fig.gca() # to get current axis
ax.add_patch(my_circle)

label = ax.annotate("2009",xy=(0,0),fontsize=30,ha='center',va='center')

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()
plt.show()


# In[20]:


# white circle with label at center
fig,ax = plt.subplots()
ax = fig.add_subplot(111)

x = 0
y = 0

circle = plt.Circle( (x,y), radius=1, color='red')

ax.add_patch(circle)

label = ax.annotate("2009", xy=(x,y),fontsize=30,ha='center')

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show()


# In[21]:


# Seat won by Alliances and Major political parties

colors = ('orange','green','red','cyan','brown','grey','blue','indigo',
          'beige','yellow','cadetblue','khaki')

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.pie(SeatsWin[2009],labels=SeatsWin['Alliance'],colors=colors,autopct='%1.1f%%')
my_circle1 = plt.Circle( (0,0),0.7,color='white')
fig = plt.gcf()
fig.suptitle('Winning Percentages by Alliances and major Political Parties',fontsize=14)
ax = fig.gca()
ax.add_patch(my_circle1)

label = ax.annotate('2009',xy=(0,0),fontsize=30,ha='center',va='center')

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()


plt.subplot(1,2,2)
plt.pie(SeatsWin[2014],labels=SeatsWin['Alliance'],colors=colors,autopct='%1.1f%%')
my_circle1 = plt.Circle( (0,0),0.7,color='white')
fig = plt.gcf()
fig.suptitle('Winning Percentages by Alliances and major Political Parties',fontsize=14)
ax = fig.gca()
ax.add_patch(my_circle1)

label = ax.annotate('2014',xy=(0,0),fontsize=30,ha='center',va='center')

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()


plt.show()


# In[23]:


# function to add value label to plot
def annot_plot(ax,w,h):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))
        
CatWin = LS0914[(LS0914.Position==1)].groupby(['Candidate Category','Year'])['Position'].sum().reset_index().pivot(index='Candidate Category', columns='Year',values='Position').reset_index().fillna(0).sort_values([2014,2009], ascending=False).reset_index(drop = True)
#print(CatWin['Candidate Category'].unique())
#CatWin
        

nx = CatWin.plot(kind='bar',title='Winning Category',figsize=(15,10),legend=True,fontsize=12)
nx.set_xlabel('Candidate Category',fontsize=12)
nx.set_ylabel('Seats Won',fontsize=12)

# Modifying Axis Labels

labels = [item.get_text() for item in nx.get_xticklabels()]
labels[0] = 'GEN'
labels[1] = 'SC'
labels[2] = 'ST'
nx.set_xticklabels(labels)

annot_plot(nx,0.05,5)


# In[25]:


CatAlliance09 = LS0914[(LS0914.Position==1) & (LS0914.Year==2009)].groupby(['Alliance','Candidate Category'])['Position'].sum().unstack().reset_index().fillna(0)
CatAlliance14 = LS0914[(LS0914.Position==1) & (LS0914.Year==2014)].groupby(['Alliance','Candidate Category'])['Position'].sum().unstack().reset_index().fillna(0)

nx = CatAlliance09.plot(kind='bar', title ="2009 Winning Category", figsize=(15, 10), legend=True, fontsize=12)
nx.set_xlabel("Candidate Category", fontsize=12)
nx.set_ylabel("Seats Won", fontsize=12)

# Modifying Axis Labels
labels = [item.get_text() for item in nx.get_xticklabels()]
labels[0:11] = CatAlliance09['Alliance']
#labels[1]= 'SC'
#labels[2]='ST'
nx.set_xticklabels(labels)

annot_plot(nx,0.05,5)

nx = CatAlliance14.plot(kind='bar', title ="2014 Winning Category", figsize=(15, 10), legend=True, fontsize=12)
nx.set_xlabel("Candidate Category", fontsize=12)
nx.set_ylabel("Seats Won", fontsize=12)

# Modifying Axis Labels
labels = [item.get_text() for item in nx.get_xticklabels()]
labels[0:11] = CatAlliance14['Alliance']
#labels[1]= 'SC'
#labels[2]='ST'
nx.set_xticklabels(labels)

annot_plot(nx,0.05,5)


# In[26]:


plt.style.use('seaborn-deep')
Age09 = LS0914[(LS0914.Position==1) & (LS0914.Year==2009)]['Candidate Age'].tolist()
Age14 = LS0914[(LS0914.Position==1) & (LS0914.Year==2014)]['Candidate Age'].tolist()

bins = np.linspace(20,90,10)
plt.hist([Age09,Age14],bins,label=['2009','2014'])
plt.legend(loc='upper right')
plt.xlabel('Age of winners in years')
plt.ylabel('Total number of winners')
plt.title('Distribution of Age of the winners')
plt.show()


# In[30]:


# Age Distribution of Winning Candidates in 2009 & 2014   NDA & UPA in India Elections

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.style.use('seaborn-deep')

Age09UPA = LS0914[(LS0914.Position==1) & (LS0914.Year==2009) & (LS0914.Alliance=='UPA')]['Candidate Age'].tolist()
Age14UPA = LS0914[(LS0914.Position==1) & (LS0914.Year==2014) & (LS0914.Alliance=='UPA')]['Candidate Age'].tolist()

Age09NDA = LS0914[(LS0914.Position==1) & (LS0914.Year==2009) & (LS0914.Alliance=='NDA')]['Candidate Age'].tolist()
Age14NDA = LS0914[(LS0914.Position==1) & (LS0914.Year==2014) & (LS0914.Alliance=='NDA')]['Candidate Age'].tolist()

bins = np.linspace(20, 90, 10)
plt.hist([Age09NDA, Age14NDA], bins, label=['2009', '2014'])
plt.legend(loc='upper right')
plt.xlabel('Age Of NDA winners in years')
plt.ylabel('Total Number of NDA winners')
plt.title('Distribution of Age of NDA winners')


plt.subplot(1,2,2)
bins = np.linspace(20, 90, 10)
plt.hist([Age09UPA, Age14UPA], bins, label=['2009', '2014'])
plt.legend(loc='upper right')
plt.xlabel('Age Of UPA winners in years')
plt.ylabel('Total Number of UPA winners')
plt.title('Distribution of Age of UPA winners')

plt.show();


# In[33]:


# Gender Distribution of Winning candidates in 2009 and 2014 india elections
colors = ['#0000CD','#CD3333']
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)

plt.pie(LS0914[(LS0914.Position==1) & (LS0914.Year==2009)]['Candidate Sex'].value_counts(),labels=['Male','Female'],autopct='%1.1f%%',colors=colors,startangle=90)
my_circle1 = plt.Circle((0,0),0.7,color='white')
fig = plt.gcf()
fig.suptitle('Gender Distribution in 2009 & 2014 India Elections',fontsize=14)
ax = fig.gca()
ax.add_patch(my_circle1)
label = ax.annotate('2009',xy=(0,0),fontsize=30,ha='center',va='center')
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()


plt.subplot(1,2,2)
plt.pie(LS0914[(LS0914.Position==1) & (LS0914.Year==2014)]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle2=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() #gcf means get current figure
ax = fig.gca() # gca means get current axis
ax.add_patch(my_circle2)

label = ax.annotate("2014", xy=(0, 0), fontsize=30, ha="center",va="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show();


# In[34]:


# Gender Distribution of Winning Candidates in 2009 - NDA vs UPA in India Elections

colors = ['#0000CD','#CD3333']
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.pie(LS0914[(LS0914.Position==1) & (LS0914.Year==2009)& (LS0914.Alliance=='NDA')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle1=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() 
fig.suptitle("Gender Distribution in 2009 - NDA vs UPA", fontsize=14) # Adding supertitle with pyplot import
ax = fig.gca() 
ax.add_patch(my_circle1)
label = ax.annotate("NDA", xy=(0, 0), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.subplot(1,2,2)
plt.pie(LS0914[(LS0914.Position==1) & (LS0914.Year==2009)& (LS0914.Alliance=='UPA')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle2=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() #gcf means get current figure
ax = fig.gca() # gca means get current axis
ax.add_patch(my_circle2)

label = ax.annotate("UPA", xy=(0, 0), fontsize=30, ha="center",va="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show();


# In[35]:


# Reading 2009 electros dataset
LS09Elec = pd.read_csv('LS2009Electors.csv')
print(LS09Elec.shape)
LS09Elec.head()


# In[36]:


# Reading 2014 electros dataset
LS14Elec = pd.read_csv('LS2014Electors.csv')
print(LS14Elec.shape)
LS14Elec.head()


# In[37]:


LS09Elec.STATE.unique()


# In[38]:


LS14Elec.STATE.unique()


# In[39]:


LS14Elec.columns


# In[41]:


# calculate poll percentage
LS09Elec = LS09Elec.groupby('STATE').mean()
LS09 = LS09Elec[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()
Y09 = [2009 for i in range(35)]
S09 = list(LS09['POLL PERCENTAGE'].keys())
P09 = list(LS09['POLL PERCENTAGE'].values())

LS14Elec = LS09Elec.groupby('STATE').mean()
LS14 = LS14Elec[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()
Y14=[2014 for i in range(35)]
S14=list(LS14['POLL PERCENTAGE'].keys())
P14=list(LS14['POLL PERCENTAGE'].values())

Data = {'YEAR':Y09+Y14,'STATE':S09+S14,'Poll_Percentage':P09+P14}
DF = pd.DataFrame(data=Data)
ax = plt.subplots(figsize=(6,20))
sns.barplot(x=DF.Poll_Percentage,y=DF.STATE,hue=DF.YEAR)
plt.title('Poll Percentage of States 2009 and 2014')


# In[ ]:




