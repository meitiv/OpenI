#!/usr/bin/env python3

import urllib.request as rq

modalities = [ 'c', 'm', 'mc', 'ph', 'u', 'p', 'x' ]

import json
base = 'https://openi.nlm.nih.gov'
baseurl = base + '/retrieve.php?query=placenta'
root = '/home/leapfrog/Projects/OpenIsearch/'
import os
for mod in modalities:
    urls = set()
    # make the directory if not already there
    try:
        os.mkdir(root + mod)
    except:
        pass
    os.chdir(root + mod)
    mn = 1
    mx = 30
    while True:
        # build url
        url = baseurl + '&n=' + str(mn) + \
              '&m=' + str(mx) + '&it=' + mod
        # get the result as a dict
        result = json.loads(rq.urlopen(url).read().decode())
        # break if reached the end
        if 'count' not in result or result['count'] == 0: break        
        # for each returned figure save its url
        for fig in result['list']: urls.add(fig['imgGrid150'])
        # increment mn and mx
        mn += 30
        mx += 30
        
    # output the urls
    with open('urls', 'w') as f: f.write('\n'.join(urls))
