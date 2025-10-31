#!/usr/bin/env python3
import requests
import urllib.parse
import re
from tqdm import tqdm
def get_article_data(article_id: int):
    """
    Fetch full article data from VNExpress GW API using the given article_id.
    """
    base_url = "https://gw.vnexpress.net/ar/get_full"
    
    # Define query parameters
    params = {
        "article_id": article_id,
        "data_select": urllib.parse.quote(
            "article_id,article_type,title,share_url,thumbnail_url,publish_time,lead,privacy,original_cate,article_category",
            safe=""
        ),
        "exclude_id": urllib.parse.quote(
            "4662602,4662536,4662665,4662634,4662635,4656425,4653241,4662807,4662809,4662473",
            safe=""
        ),
        "thumb_size": urllib.parse.quote("680x408,500x300,300x180", safe=""),
        "thumb_quality": 100,
        "thumb_dpr": urllib.parse.quote("1,2", safe=""),
        "thumb_fit": "crop"
    }

    # Compose final URL
    url = f"{base_url}?article_id={article_id}&data_select={params['data_select']}&exclude_id={params['exclude_id']}&thumb_size={params['thumb_size']}&thumb_quality={params['thumb_quality']}&thumb_dpr={params['thumb_dpr']}&thumb_fit={params['thumb_fit']}"

    # Send request
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        try:
            data = response.json()
            return data
        except ValueError:
            print("Error: Response is not valid JSON.")
            return None
    else:
        print(f"Error: HTTP {response.status_code}")
        return None

def strip_html_tags_regex(html_string):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', html_string)


id_list_by_type = {
    "đời sống": ["4946014", "4933833", "4925297", "4933950", "4933666", "4933957", "4934213", "4934222", "4934307", "4933946", "4934039", "4934121", "4934718", "4934499", "4934631", "4934563", "4927471"],
    "du lịch": ["4934688", "4934446", "4934516", "4934125", "4933860", "4934570", "4926775", "4935138", "4934912", "4934909", "4934787", "4934782", "4931854", "4935002", "4934764"],
    "khcn": ["4934387", "4934403", "4935226", "4935205", "4934918", "4933963", "4934003", "4933925", "4932604", "4934068"],
}

def construct_test_file():
    with open("test_articles.csv", "w", encoding="utf-8") as f:
        f.write("no,type,id,title,lead,content\n")
        no = 1
        for type_name, id_list in id_list_by_type.items():
            for article_id in tqdm(id_list):
                data = get_article_data(int(article_id))
                if data and 'data' in data:
                    title = data['data'].get('title', '').replace('"', '""')
                    lead = data['data'].get('lead', '').replace('"', '""')
                    content = repr(strip_html_tags_regex(data['data'].get('content', '')).replace('"', '""'))
                    f.write(f'{no},{type_name},{article_id},"{title}","{lead}","{content}"\n')
                    no += 1

if __name__ == "__main__":
    construct_test_file()