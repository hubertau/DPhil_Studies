"""
Quick script to show the rate limits left for an account
"""

import requests
import json
from . import constants

def main():
    url = 'https://api.mediacloud.org/api/v2/auth/profile'
    response = requests.get(url=url, params={
        "key": constants.API_KEY
    })
    print(json.dumps(response.json(),indent=4))

if __name__ == "__main__":
    main()
