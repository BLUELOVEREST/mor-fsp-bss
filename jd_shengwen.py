import requests
import json
import time
import uuid
import hashlib
from urllib.parse import urlencode
import sys
import importlib
importlib.reload(sys)

true = True
false = False

url = 'https://aiapi.jd.com/jdai/vpr'

encode = {
  'channel':1,
  'format':'wav',
  'sample_rate':16000
  }
Property = {
  'platform':'Linux&Centos&7.3',
  'version':'0.0.0.1',
  'vpr_mode':'test',
  'autoend':False,
  'encode':encode,
  }

headers = {
  'Content-Type':'application/octet-stream',
  'Application-Id':'123456789',
  'Request-Id':str(uuid.uuid1()),
  'User-Id':'zzc', # your user-id
  'Sequence-Id':str(-1),
  'Server-Protocol':str(1),
  'Net-State':str(2),
  'Applicator':str(1),
  'Property':json.dumps(Property)
  }
query = {
    # 'appkey':'233a0dc48a864c1fefe21c0526414a99', #niukun
    # 'appkey':'cc8a5f69db9ea743b2e1fc1ea618135b', #zzc
    'appkey':'7312b709532b1bf0464f796355ce092a', #xiaozhao
    'timestamp':'1607941657',
    'sign':''
    }

# audiofile = '15.wav' #change to your audio file
# secretkey = 'c04c1f411d66000c5872625474971540' #niukun
# secretkey = '0d146696273e86adbc62227ca4f85479' #zzc
secretkey = 'e5a011c17f19267933bef142d24eaf9f'#xiaozhao


def test_single(audiofile):
  query['timestamp'],query['sign'] = sign(secretkey)
  url_query = '?'.join([url, urlencode(query)])
  #seq = 1
  #packagelen = 4000

  with open(audiofile, mode='rb') as f:
    while True:
      audiodata=f.read()
      #audiodata=f.read(int(packagelen))
      if not audiodata:
        break
      #else:
        #if len(audiodata) < int(packagelen):
          #headers['Sequence-Id'] = str(-seq)
        #else:
          #headers['Sequence-Id'] = str(seq)
      r = requests.post(url_query, headers=headers, data=audiodata)
      #seq += 1
  # print(r.text)
  result_info = r.text
  result_dict = eval(result_info)
  final_result = result_dict['result']['result'][0]['text']
  if final_result == "accept" :
    print("1")
  else:
    print("0")


def sign(secretkey):
    m = hashlib.md5()
    nowTime = int(time.time() * 1000)
    before = secretkey + str(nowTime)
    m.update(before.encode('utf8'))
    return str(nowTime), m.hexdigest()

if __name__ == '__main__':
    audiofile = sys.argv[1] if len(sys.argv) > 1 else audiofile
    test_single(audiofile)