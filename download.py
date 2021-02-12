import gdown

url = 'https://drive.google.com/uc?export=download&id=1Y5_dO2KaaJYvAJbvajKk_GNRbxr_0f6R'
output = 'data.tar.gz'
gdown.download(url,output,quiet=False)
