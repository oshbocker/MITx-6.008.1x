# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:47:48 2016

@author: mysterion
"""

def SECdownload(year, month):
    root = None
    feedFile = None
    feedData = None
    good_read = False
    itemIndex = 0
    edgarFilingsFeed = 'http://www.sec.gov/Archives/edgar/monthly/xbrlrss-' + str(year) + '-' + str(month).zfill(2) + '.xml'
    print(edgarFilingsFeed)
    if not os.path.exists("edgar/" + str(year)):
        os.makedirs("edgar/" + str(year))
    if not os.path.exists("edgar/" + str(year) + '/' + str(month).zfill(2)):
        os.makedirs("edgar/" + str(year) + '/' + str(month).zfill(2))
    target_dir = "edgar/" + str(year) + '/' + str(month).zfill(2) + '/'
    try:
        feedFile = urlopen(edgarFilingsFeed)
        try:
            feedData = feedFile.read()
            good_read = True
        finally:
            feedFile.close()
    except HTTPError as e:
        print("HTTP Error:", e.code)
        
