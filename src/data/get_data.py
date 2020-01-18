# author: Karanpal	Singh, Sreejith	Munthikodu, Sirine	Chahma
# date: 2020-01-17

'''This script downloads the data from a given url and saves in the data
folder in the project directory. This script takes a url to the data and a 
file location as the arguments.

Usage: get_data.py --url=<url> --file_location=<file_location>
 
'''

import requests
from docopt import docopt

opt = docopt(__doc__)

def main(url, file_location):
    # download and save data
    r = requests.get(url)
    with open(file_location, "wb") as f:
        f.write(r.content) 
    print(f"file successfully saved to {file_location}")

if __name__ == "__main__":
    main(opt["--url"], opt["--file_location"])