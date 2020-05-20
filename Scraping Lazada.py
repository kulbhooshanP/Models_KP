import time, re
tic = time.clock()
from selenium import webdriver
import  pandas as pd
from bs4 import BeautifulSoup

###########################################################################
url='http://www.lazada.com.ph/pilaten-black-head-remover-60g-12598827.html?spm=a2o4l.category-080200000000.0.0.cPgghE&ff=1&sc=EQMV'

Path="D:/Users/kulpatil/Desktop/"

OutputfileName="Lazada_output.xlsx"

driver = webdriver.Chrome("/Python Scripts/chromedriver_win32/chromedriver.exe")
############################################################################

postsaleslist=[]

driver.get(url)

time.sleep(5)

Ipagesource=driver.page_source

soup = BeautifulSoup(Ipagesource)

counter=1

create_xpath=''

a = soup.find_all(class_ = 'c-review__content')

   

for la in a:
    selected_field= str(la.text.strip())

c = soup.find_all(class_ = 'c-rating-total__text-total-review')
for laa in c:
    No_of_Reviews=[int(s) for s in laa.text.strip().split() if s.isdigit()][1]
   
if No_of_Reviews % 10 == 0:
    No_of_Pages=int(No_of_Reviews/10)
else:
    No_of_Pages=int(No_of_Reviews/10)+1

#print "[Title, Ratings, Days ago, Username, Text]"
for foo in a:
    title = foo.find('div', attrs={'class': 'c-review__title'})
    date = foo.find('div', attrs={'class': 'c-review__date'})
    rati = foo.find_all('div', attrs={'class': 'c-rating-stars__active'})    
    ratings=re.findall('\d+', str(rati))
    ratings=int(''.join(ratings))
    if ratings==None or ratings==0:
        ratings=0
    else:    
        Iratings=ratings/20
    userinfo = foo.find_all('div', attrs={'class': 'c-review__user-info'})
    
    for i in userinfo:
        username=i.find(attrs={'class': 'c-review__name-text'}).text.strip()
    comment=foo.find(attrs={'class': 'c-review__comment'})
    
    if comment== None:
        Icomment=''
    else:
        Icomment=comment.text.strip()
    
    if title== None:
        Ititle=''
    else:
        Ititle=title.text.strip()

    if date== None:
        Idate=''
    else:
        Idate=date.text.strip()
    
    if username== None:
        Iusername=''
    else:
        Iusername=username
    temp=[url, Ititle, Iratings , Idate,  Iusername, Icomment, create_xpath, selected_field]
    print temp
    postsaleslist.append(temp)


if No_of_Pages>1:
    for ii in range(3,No_of_Pages+2):
        try:
            time.sleep(3)
            if int(selected_field)>=4:
                create_xpath='//*[@id="reviewslist"]/div[3]/div/div/ul/li[5]/span'
            else:
                create_xpath='//*[@id="reviewslist"]/div[3]/div/div/ul/li['+str(ii)+']/span'
            print (create_xpath)
            elem_load_more=driver.find_element_by_xpath(create_xpath)
            elem_load_more.click()
            time.sleep(3)
            Ipagesource=driver.page_source 
            soup = BeautifulSoup(Ipagesource)
            counter=counter+1
            a = soup.find_all(class_ = 'c-review__content')
            #print "[Title, Ratings, Days ago, Username, Text]"
            b = soup.find_all(class_ = 'c-review-paging__link c-review-paging__link_state_selected')
            for la in b:
                selected_field= str(la.text.strip())
            for foo in a:
                title = foo.find('div', attrs={'class': 'c-review__title'})
                date = foo.find('div', attrs={'class': 'c-review__date'})
                rati = foo.find_all('div', attrs={'class': 'c-rating-stars__active'})    
                ratings=re.findall('\d+', str(rati))
                ratings=int(''.join(ratings))
                if ratings==None or ratings==0:
                    ratings=0
                else:    
                    Iratings=ratings/20
                userinfo = foo.find_all('div', attrs={'class': 'c-review__user-info'}) 
                for i in userinfo:
                    username=i.find(attrs={'class': 'c-review__name-text'}).text.strip()
                comment=foo.find(attrs={'class': 'c-review__comment'})
                
                if comment== None:
                    Icomment=''
                else:
                    Icomment=comment.text.strip()
                if title== None:
                    Ititle=''
                else:
                    Ititle=title.text.strip()
                if date== None:
                    Idate=''
                else:
                    Idate=date.text.strip()
                if username== None:
                    Iusername=''
                else:
                    Iusername=username
                temp=[url, Ititle, Iratings , Idate,  Iusername, Icomment, create_xpath, selected_field]
                print temp
                postsaleslist.append(temp)
        except Exception as inst:
            print ("Rerun")
                
Output=pd.DataFrame(postsaleslist,columns=['Url','Title', 'Ratings', 'Days ago', 'Username', 'Text', 'xpath', 'Page Number'])

writer = pd.ExcelWriter(Path+OutputfileName)

Output.to_excel(writer,'Sheet1', index=False)

writer.save()

print ("URLs extracted !!!", time.clock() - tic)

print ("Processing Time:", time.clock() - tic)

driver.close()

driver.quit()

#########################################################EOF####################

