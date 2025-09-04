import threading
import requests
import json
import io

import os
from PIL import Image

def split_dict(dict,split_num):
    dict_length = len(dict)
    split_size = dict_length // split_num + (dict_length % split_num > 0)
    result = []

    for i in range(0,dict_length,split_size):
        result.append({k:dict[k] for k in list(dict.keys())[i:i+split_size] })

    return result


# down load task
class Downloader(threading.Thread):
    def __init__(self, url_dict):
        super().__init__()
        self.url_dict = url_dict
        self.path = './images'
        self.fail_list = []
        #self.all_download_url = []

    def run(self):
        for key in self.url_dict.keys():
            url_list = url_dict[key]

            # preprocessing url
            for idx,origin_url in enumerate(url_list ):
                self.download(key,idx,origin_url)

    def download(self,search_key, idx,image_url):
        obj = search_key.split(' ')[-1]

        search_key = search_key.replace(' ','_')
        save_path = os.path.join(self.path,obj)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filename = "%s_%s.%s"%(search_key,str(idx),'jpg')
                                
        image_path = os.path.join(save_path, filename)

        if not os.path.exists(image_path):
            try:
                # overlook the existed images
                image = requests.get(image_url,timeout=5)
                if image.status_code == 200:
                    with Image.open(io.BytesIO(image.content)) as image_from_web:
                        try:
                            print(
                                f"[INFO] {search_key} \t {idx} \t Image saved at: {image_path}   [url]: {image_url}  ")
                            image_from_web.save(image_path)
                        except OSError:
                            rgb_im = image_from_web.convert('RGB')
                            rgb_im.save(image_path)
                        image_from_web.close()
            except Exception as e:
                print("[ERROR] Download failed: ", search_key,image_url)
                self.fail_list.append(f"{search_key} : {image_url}")
                pass



if __name__ == "__main__":
    url_json_path ='./all_composition_url.json'

    url_dict = {}
    with open(url_json_path,'r') as f:
        url_dict = json.load(f)


    # Split the URLs evenly among threads
    num_threads = 16
    
    split_url_dict = split_dict(url_dict,num_threads)


    # downloader = Downloader(split_url_dict[0])
    # downloader.run()
    threads = []
    for chunk in  split_url_dict:
        downloader = Downloader(chunk)
        downloader.start()
        threads.append(downloader)
        

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    with open('fail.log','w') as f:
        for d in threads:
            for fail_url in d.fail_list:
                f.write(fail_url+'\n')