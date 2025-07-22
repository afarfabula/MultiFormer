# 引入模块
from obs import ObsClient, PutObjectHeader
import os

# 创建ObsClient实例
obsClient = ObsClient(
    access_key_id='UXC42LP5RRWVWSCHBWRT',     # 替换成你的Access Key ID
    secret_access_key='W80HelKVRo4Gdpc93rWLPbnPnJGdT00Eb6MOtgg7',  # 替换成你的Secret Access Key
    server='obs.cn-north-4.myhuaweicloud.com'   # 替换成你的服务器地址
)

# 定义上传文件的函数
def upload_file(client, bucket_name, local_path, obs_path, headers=None):
    try:
        resp = client.putFile(
            bucket_name,
            obs_path,
            local_path,
            headers=headers
        )
        if resp.status < 300:
            resp.status=resp.status
            print('Upload successful for:', obs_path)
            print('objectUrl:', resp.body.objectUrl)
            #print('requestId:', resp.requestId)
            #print('etag:', resp.body.etag)
            #print('versionId:', resp.body.versionId)
            #print('storageClass:', resp.body.storageClass)
        else:
            print('Failed to upload:', obs_path)
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
    except Exception as e:
        print('Error uploading', obs_path, 'with exception:', e)

def updatepic():
    upload_file(
        obsClient,
        'obs1111112',
        'untitled1.txt',
        'PIC/untitled1.txt',
        headers=PutObjectHeader()  # 可以设置headers，例如 contentType
    )
    obsClient.close()
    return 1

if __name__ == "__main__":
    updatepic()
    updatepic()
    updatepic()
    updatepic()
