def try_field(tweet_result_obj, field_name):

    if field_name == 'hashtags':
        
        # not checking if 'entities' is in tweet result object because by stipulation
        # search query demands at least one hashtag
        hashtag_list = [i['tag'] for i in tweet_result_obj['entitites']['hashtags']]
        hashtag_list = ';'.join(hashtag_list)
        return hashtag_list
    
    elif field_name in ['full_name', 'country', 'country_code', 'name', 'place_type']:
        if 'includes' in tweet_result_obj:
            if 'places' in tweet_result_obj['includes']:
                try:
                    return tweet_result_obj['includes']['places'][field_name]
                except:
                    return 'NA'
    
    elif field_name == 'geo_id':
        if 'includes' in tweet_result_obj:
            if 'places' in tweet_result_obj['includes']:
                try:
                    return tweet_result_obj['includes']['places']['id']
                except:
                    return 'NA'
    
    elif field_name == 'bbox':
        if 'includes' in tweet_result_obj:
            if 'places' in tweet_result_obj['includes']:
                try:
                    bbox = tweet_result_obj['includes']['places']['geo']['bbox']
                    bbox = ';'.join(bbox)
                    return bbox
                except:
                    return 'NA'

    elif field_name in ['media_key', 'preview_image_url', 'type']:
        if 'includes' in tweet_result_obj:
            if 'media' in tweet_result_obj['includes']:
                try:
                    return tweet_result_obj['includes']['media'][field_name]
                except:
                    return 'NA' 

    else:
        try:
            return tweet_result_obj[field_name]
        except:
            return 'NA'



# store results in csv format
results_header = (
    'author_id',
    'text',
    'created_at',
    'id',
    'conversation_id',
    'created')
results_writer.writerow(results_header)
for tweet_result in json_response['data']:
    result_row = (
            try_field(tweet_result,'author_id'),
            try_field(tweet_result,'text'),
            try_field(tweet_result,'created_at'),
            try_field(tweet_result,'id'),
            try_field(tweet_result,'conversation_id'),
            try_field(tweet_result,'hashtags'),
            try_field(tweet_result,'')
        )       
    results_writer.writerow(result_row)



for tweet_result in json_response['data']:
    result_row = (
        try_field(tweet_result,'author_id'),
        try_field(tweet_result,'text'),
        try_field(tweet_result,'created_at'),
        try_field(tweet_result,'id')
    )
    results_writer.writerow(result_row)