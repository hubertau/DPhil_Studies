#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:14:30 2020

@author: Patrick
"""

import pandas as pd
import requests
import numpy as np
import json
import os

def query(request, lang='en'):
    request['action'] = 'query'
    request['format'] = 'json'
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get('https://%s.wikipedia.org/w/api.php' %lang, params=req).json()
        if 'error' in result:
            print('erroorrrr')
            print(result['error'])
            #raise Error(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            yield result['query']
        if 'continue' not in result:
            break
        lastContinue = result['continue']

#        print(result)

def parse(request, lang='en'):
    request['action'] = 'parse'
    request['format'] = 'json'
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get('https://%s.wikipedia.org/w/api.php' %lang, params=req).json()
        #print(result)
        if 'error' in result:
            print('erroorrrr')
            print(result['error'])
            #raise Error(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'parse' in result:
            yield result['parse']
        if 'continue' not in result:
            break
        lastContinue = result['continue']
        
def chunks(l, n):
# For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]
        
def transformt(d):
    d1 = {k:v['damaging'].get('score', None) for k, v in d.items()}
    d2 = {k:v['goodfaith'].get('score', None) for k, v in d.items()}
    d3 = {k:v['articlequality'].get('score', None) for k, v in d.items()}

    return [{k:{**{'prediction':v['prediction']}, **v['probability']} for k, v in x.items() if v} for x in [d1,d2,d3]]
        
#articles and the range(s) of dates
artranges = {'List_of_incidents_of_xenophobia_and_racism_related_to_the_2019–20_coronavirus_pandemic':[['2020-01-01T00:00:00Z', 'now']],
             'Talk:List_of_incidents_of_xenophobia_and_racism_related_to_the_2019–20_coronavirus_pandemic':[['2020-01-01T00:00:00Z', 'now']]}
savepath = ''

for n, (article, v) in enumerate(artranges.items()):
    if os.path.exists('%s%s.h5' %(savepath, article.replace('/', ':'))):
        continue

    masterdf = pd.DataFrame(columns = ['timestamp', 'revid', 'user', 'anon', 'size', 'comment', 'minor','tags', 'userhidden', 'commenthidden', 'slots'])    
    artdf = pd.DataFrame(columns = ['timestamp', 'revid', 'user', 'anon', 'size', 'comment', 'minor','tags', 'userhidden', 'commenthidden', 'slots'])
    print(article, n/len(artranges))
    print('getting edit data')
    
    for dates in v:
        try:                     
            params = {'prop':'revisions', 'titles':article, 'rvprop':'timestamp|ids|user|size|tags|flags|comment|content', 'rvlimit':'max', 'rvend':dates[0], 'rvstart':dates[1], 'rvslots':'*'}
            for i in query(params):
                for j in i['pages'].values():
                    try:
                        artdf = artdf.append(pd.DataFrame(j['revisions']).sort_values(by='timestamp'), ignore_index=True, sort=False)
                    except Exception as ex:
                        print('error', ex,  article)            
            artdf[['anon','minor', 'userhidden', 'commenthidden']] = artdf[['anon','minor', 'userhidden', 'commenthidden']].replace('', True).fillna(False)
            artdf['article'] = article.replace(' ', '_') 
            masterdf = masterdf.append(artdf, sort=False)

        except KeyboardInterrupt:
            print('error',  article)
            raise
        except Exception as ex:
            print('error', ex,  article)
    try:
        masterdf = masterdf.astype({'anon': bool, 'article': str, 'comment': str, 'commenthidden': bool, 'slots':object, 'minor': bool, 'parentid': int, 'revid': int, 'size': int, 'tags': object, 'timestamp': str, 'user': str, 'userhidden': bool})
        masterdf.to_hdf('%s%s.h5' %(savepath, article.replace('/', ':')), key='df')
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print('writing error', ex, article)

    
    # Get ORES scores
    print('getting ores scores')    
    revidlist = list(masterdf['revid'].astype(str))
    
    damagingdf = pd.DataFrame(columns = ['prediction', 'true', 'false'])
    goodfaithdf = pd.DataFrame(columns = ['prediction', 'true', 'false'])
    qualitydf = pd.DataFrame(columns = ['prediction', 'Stub', 'Start', 'C', 'B', 'GA', 'FA'])
    cl = list(chunks(revidlist, 50))
    for n, i in enumerate(cl):
        print(n/len(cl))
        revids = '|'.join(i)
        aq = requests.get('https://ores.wikimedia.org/v3/scores/enwiki?models=articlequality|damaging|goodfaith&revids=%s' %revids).json()
        
        parsed = transformt(aq['enwiki']['scores'])
        pd.DataFrame.from_dict(parsed[0], orient='index')
        damagingdf = damagingdf.append(pd.DataFrame.from_dict(parsed[0], orient='index'), sort=False)
        goodfaithdf = goodfaithdf.append(pd.DataFrame.from_dict(parsed[1], orient='index'), sort=False)    
        qualitydf = qualitydf.append(pd.DataFrame.from_dict(parsed[2], orient='index'), sort=False)    
        
        damagingdf.index = damagingdf.index.astype(int)
        goodfaithdf.index = goodfaithdf.index.astype(int)
        qualitydf.index = qualitydf.index.astype(int)
        
        #save hdfs
        for d in [damagingdf, goodfaithdf, qualitydf]:
            d['prediction'] = d['prediction'].astype(bool)        
        damagingdf.to_hdf('%s%s_damaging.h5' %(savepath, article), 'df')
        goodfaithdf.to_hdf('%s%s_goodfaith.h5' %(savepath, article), 'df')    
        qualitydf.to_hdf('%s%s_quality.h5' %(savepath, article), 'df')    

        
