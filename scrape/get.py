from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm
import os

def downloadNGAlist(basedir=''):
    print("Fetching NGA list...")
    with open(basedir + 'nga-list.html', "wb") as file:
        # get request
        response = get('http://seshatdatabank.info/data/')
        # write to file
        file.write(response.content)

def download(file_name, basedir):
    url = 'http://seshatdatabank.info/data/' + file_name
    file_name = basedir + 'ngas/' + file_name
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

def getPolities(basedir=''):
    with open(basedir + "nga-list.html") as fp:
        soup = BeautifulSoup(fp, 'html.parser')

    urls = []
    for nga in soup.findAll(attrs={'class': 'list-group-item'}):
        urls.append(nga.get('href'))

    print('Downloading polity info...')
    for url in tqdm(urls):
        download(url, basedir)

if __name__ == '__main__':
    if not os.path.isfile('nga-list.html'):
        downloadNGAlist()
    getPolities()
