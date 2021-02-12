import gdown

url = 'https://drive.google.com/uc?export=download&id=1cBIbXLcKb4ZkD6wvyMVg98XT2SnOdqWG'
output = 'data.tar.gz'
gdown.download(url,output,quiet=False)
