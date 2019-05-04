from bs4 import BeautifulSoup
import os
import csv

def unpack(tup):
    ((code,name),date) = tup
    return [code,name,date]


def parsePolities(basedir=''):
    print('Scrubbing polity info...')
    ngas = []
    for filename in os.listdir(basedir + 'ngas/'):
        with open(basedir + "ngas/" + filename) as fp:
            soup = BeautifulSoup(fp, 'html.parser')

        newNgas = []
        for nga in soup.findAll(attrs={'class':'nga-name'}):
            try:
                newNgas.append((nga.a.get('href').split('-')[-1],nga.a.contents[0]))
            except:
                continue


        dates = []
        for ngaDate in soup.findAll(attrs={'class':'variable-name'}):
            dates.append(ngaDate.contents[0])

        n = list(zip(newNgas, dates[-1*len(newNgas):]))
        ngas += list(map(unpack, n))



    with open(basedir + 'nameDates.csv','w') as f:
        f.write('Polity,Polity_name,Era\n')
        writer = csv.writer(f)
        writer.writerows(ngas)

if __name__ == '__main__':
    parsePolities()
