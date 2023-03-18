"""Volume 3: Web Scraping.
Jane Slagle
Vol 3 lab
1/17/23
"""

import requests
from bs4 import BeautifulSoup
import re
from matplotlib import pyplot as plt
import os
import numpy as np

# Problem 1
def prob1():
    """Use the requests library to get the HTML source for the website 
    http://www.example.com.
    Save the source as a file called example.html.
    If the file already exists, do not scrape the website or overwrite the file.
    """
    if not os.path.exists("./example.html"):               #this only runs the code if the file does not already exists
        response = requests.get("http://www.example.com")  #get HTML source for website
    
        source = open('example.html', 'w')  #write to example.html file
        source.write(response.text)         #save source as example.html file, html source is in text attribute, so actually write html source into file here
        source.close()                      #close file since used only an open and not a with open to write to the file
    
# Problem 2
def prob2(code):
    """Return a list of the names of the tags in the given HTML code.
    Parameters:
        code (str): A string of html code
    Returns:
        (list): Names of all tags in the given code"""
    soup_object = BeautifulSoup(code, 'html.parser')  #get Beautiful soup object with our code so that can use it
    
    #use find_all() beautiful soup method to return list of all tags in html code that used to make beautiful soup object:
    all_tags = soup_object.find_all(True)
    
    #use name attribute to get list of names of tags in all_tags list:
    tag_names = [all_tags[i].name for i in range(len(all_tags))]
    
    return tag_names
    
# Problem 3
def prob3(filename="example.html"):
    """Read the specified file and load it into BeautifulSoup. Return the
    text of the first <a> tag and whether or not it has an href
    attribute.
    Parameters:
        filename (str): Filename to open
    Returns:
        (str): text of first <a> tag
        (bool): whether or not the tag has an 'href' attribute
    """
    #read inputted file in:
    read_file = open(filename, 'r')
    data = read_file.read()
    
    #load into BeautifulSoup:
    file_soup = BeautifulSoup(data, "html.parser") #do same as ex box above prob bc working with an html here too
    
    read_file.close()  #need to close the file since used open to read in the file
    
    #find first <a> tag:
    first_a_tag = file_soup.a #only 1st tag of each name is accessible directly from pig_soup so doing this gives the 1st a tag just like how we want!
    
    text = first_a_tag.get_text()  #return text of 1st <a> tag using get_text attribute method
    
    #check if has href attribute (make boolean value indicating whether has it or not):
    hyper = False     #initialize it as False
    if hasattr(first_a_tag, "href"):
        hyper = True  #change it to be true if has the attribute
        
    return text, hyper

# Problem 4
def prob4(filename="san_diego_weather.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the following tags:

    1. The tag containing the date 'Thursday, January 1, 2015'.
    2. The tags which contain the links 'Previous Day' and 'Next Day'.
    3. The tag which contains the number associated with the Actual Max
        Temperature.

    Returns:
        (list) A list of bs4.element.Tag objects (NOT text).
    """
    #read in "san_diego_weather.html" and load into BeautifulSoup: do exact same way as did in prob 3
    read_file = open(filename, 'r')
    data = read_file.read()
    
    #load into BeautifulSoup:
    file_soup = BeautifulSoup(data, "html.parser") #do same as ex box above prob bc working with an html here too
    
    read_file.close()  #need to close the file since used open to read in the file
    
    #get tag containing date "Thursday, Jan 1, 2015":
    ans_1 = file_soup.find(string = "Thursday, January 1, 2015").parent  #the result is actual string so go up 1 level (doing .parent) to get the tag
    
    #get tag containing links "Previous Day" and "Next Day":
    ans_2_1 = file_soup.find(string = re.compile("Previous Day")).parent #.parent gets the actual tag, need re.compile the string bc of super cool regex stuff woohoo
    ans_2_2 = file_soup.find(string = re.compile("Next Day")).parent  
    
    #get tag that has 59 in it:
    #can do firefox san_diego_weather.html in terminal, find from website that max temp is 59, its associated w/ 'Max Temperautre' string
    #open file w/ nano, navigate to Max Temperature text
    #do .parent goes to span tag, .parent again goes to 56, but dont want 56, want 59 so do .next_sibling goes to <td>, .next_sibling again goes to line w/ 59 on it
    #.span goes into the actual tag and .span again goes inside span to actually get 59
    ans_3 = file_soup.find(string = "Max Temperature").parent.parent
    ans_3 = ans_3.next_sibling.next_sibling
    ans_3 = ans_3.span.span
    
    #return list w/ all tags just got
    return [ans_1, ans_2_1, ans_2_2, ans_3]

# Problem 5
def prob5(filename="large_banks_index.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the tags containing the links to bank data from September 30, 2003 to
    December 31, 2014, where the dates are in reverse chronological order.

    Returns:
        (list): A list of bs4.element.Tag objects (NOT text).
    """
    #read in large_banks_index.html:
    read_file = open(filename, 'r')
    data = read_file.read()
    
    #load into BeautifulSoup:
    file_soup = BeautifulSoup(data, "html.parser") #do same as ex box above prob bc working with an html here too
    
    read_file.close()  #need to close the file
    
    #get all of strings that contain dates from Sep 30 2003-Dec 31 2014, it's enough to only have the year to get all of the dates
    all_dates = file_soup.find_all(string = re.compile("20(0[3-9]|1[0-4])"))  #use a regex expression to cover all of the years want
    
    #but want all of the TAGS not the strings so loop through all of the dates just found and get the tag for each of them by doing .parent
    all_tags = []
    for date in all_dates:
        all_tags.append(date.parent)
    
    return all_tags[:-2]  #the last 2 dates in all_dates are out of the range that we want so don't include them in answer

# Problem 6
def prob6(filename="large_banks_data.html"):
    """Read the specified file and load it into BeautifulSoup. Create a single
    figure with two subplots:

    1. A sorted bar chart of the seven banks with the most domestic branches.
    2. A sorted bar chart of the seven banks with the most foreign branches.

    In the case of a tie, sort the banks alphabetically by name.
    """
    #read in large_banks_data.html:
    read_file = open(filename, 'r')
    data = read_file.read()
    
    #load into BeautifulSoup:
    file_soup = BeautifulSoup(data, "html.parser") #do same as ex box above prob bc working with an html here too
    
    read_file.close()  #need to close file
    
    tags = file_soup.find_all("tr")  #all bank tags have a tr thing so doing this will help get all bank tags
    tags = tags[5:-10]               #1st 5 and last 10 are a bunch of garbage so dont include them, can see that when print them to terminal
    
    names = []    #lists to store all of the tags in for the names, domestic and foreign branches
    domest = []
    fore = []
   
    #do .string bc only care about text, dont care about tags
    for tag in tags:  #.contents gives list of all children
        names.append(tag.contents[1].string)    #if print out tags.contents see that the name is always 2nd thing in the list of children tags so have index 1
        dom_thing = tag.contents[19].string  #same thing as before, count to see where domestic number is in list, able to see from doing firefox filename in terminal
        for_thing = tag.contents[21].string
        
        #need to get rid of the commas and periods that they have in domestic and foreign columns:
        if "," in dom_thing:
            dom_thing = dom_thing.replace(",",'')
            
        if "." in dom_thing:
            dom_thing = dom_thing.replace(".","0")  #replace w/ 0 so that when convert it, its a number
            
        domest.append(dom_thing)   #only append after gotten rid of the commas and periods
        
        if "," in for_thing:
            for_thing = for_thing.replace(",",'')
        
        if "." in for_thing:
            for_thing = for_thing.replace(".","0")
            
        fore.append(for_thing)
    
    #now turn each list into np array so that can sort them in order of greatest to smallest (cant do w/ a list): each list elements are strings, but need to be floats
    #in order to actually sort them
    names_array = np.array(names)
    domest_array = np.array(domest, dtype = int)  #have to convert to ints so that able to argsort it (otherwise it will be strings)
    fore_array = np.array(fore, dtype = int)
    
    #now actually sort them as want and get the 1st 7
    mask = np.flip(np.argsort(domest_array))  #argsort sorts as smallest to greatest so do np.flip to reverse order of all entries, gives largest indices for domestic branches
    domest_names_array = names_array[mask][:7]  #now get names of the 7 banks w/ most domestic branches
    
    #now do the same thing but w/ foreign branches:
    mask2 = np.flip(np.argsort(fore_array))
    fore_names_array = names_array[mask2][:7]
   
    #now plot the 2 subplots:
    plt.subplot(1,2,1)
    #plot 7 banks w/ most domestic branches: this is domest_names_array
    plt.barh(domest_names_array, domest_array[mask][:7], color = "palevioletred")
    plt.yticks(fontsize = 6)
    plt.title("7 banks with most domestic branches", color = "navy")
    
    plt.subplot(1,2,2)
    #plot 7 banks w/ most foreign branches: this is fore_names_array
    plt.barh(fore_names_array, fore_array[mask2][:7], color = "mediumpurple") 
    plt.yticks(fontsize = 6)
    plt.title("7 banks with most foreign branches", color = "navy")
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    
    #test prob 1:
    #print(prob1())  #func doesn't return anything so should get None out
    
    #test prob 2:
    #string of HTML code that will test prob with:
    small_example_html = """
    <html><body><p>
    Click <a id='info' href='http://www.example.com'>here</a> for more information.
    </p></body></html>
    """
    #print(prob2(small_example_html))  #want it to return list of names of tags in code from HTML string. from example box in lab manual: return html, body, p and a
    #all tags are anything surrounded with <>, so each tag is in format: <tag_name>
    
    #test prob 3:
    #print(prob3())
    
    #test prob 4:
    #print(prob4())
    
    #test prob 5:
    #print(prob5())
    
    #test prob 6:
    #print(prob6())
    
    pass
