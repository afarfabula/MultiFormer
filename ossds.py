import pexpect
import re
import time
import threading
import zipfile
import os
from datetime import datetime
from pathlib import Path

# 文件路径配置
TIMESTAMP_FILE = "latest_timestamp.txt"
DOWNLOAD_DIR = "/home/qyy/notebooks/dynamicdatasets/"
EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "extracted")  # 解压目录

class UnzipWorker(threading.Thread):
    """独立线程的Worker，负责递归解压所有嵌套ZIP文件"""
    def __init__(self, zip_path):
        super().__init__()
        self.zip_path = zip_path
        self.daemon = True  # 主线程退出时自动终止Worker线程

    def run(self):
        try:
            self._recursive_unzip(self.zip_path)
            #print(f"[解压完成] 所有嵌套ZIP文件已处理: {self.zip_path}")
        except Exception as e:
            print(f"[Worker错误] 解压失败: {e}")

    def _recursive_unzip(self, zip_path):
        """递归解压ZIP文件，包括嵌套的ZIP"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 创建以时间戳命名的子目录
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            extract_path = os.path.join(EXTRACT_DIR, f"unzipped_{timestamp}")
            os.makedirs(extract_path, exist_ok=True)
            
            # 首次解压
            zip_ref.extractall(extract_path)
            print(f"[首次解压] 文件解压到: {extract_path}")

            # 检查解压后的目录中是否仍有ZIP文件
            for root, _, files in os.walk(extract_path):
                for file in files:
                    if file.endswith('.zip'):
                        nested_zip = os.path.join(root, file)
                        self._recursive_unzip(nested_zip)  # 递归解压嵌套ZIP

def run_check(task):
    """检查OSS文件列表并返回最新的temp文件时间戳"""
    try:
        child = pexpect.spawn(task, encoding='utf-8', timeout=30)
        child.expect("File number is", timeout=60)
        output = child.before

        # 提取时间戳
        timestamps = re.findall(r"oss://csidatasets/batch_(\d{14})\.zip", output)
        if not timestamps:
            print("[警告] 未找到任何 temp_*.zip 文件")
            return None
        
        latest_timestamp = max(timestamps)
        print(f"最新 temp 文件时间戳: {latest_timestamp}")
        return latest_timestamp

    except pexpect.EOF:
        print("[错误] 命令中断，检查网络或权限")
    except pexpect.TIMEOUT:
        print("[错误] 操作超时")
    finally:
        if 'child' in locals() and child.isalive():
            child.close()

def run_download(timestamp):
    """下载指定时间戳的ZIP文件并触发解压"""
    try:
        if not timestamp:
            return False
        
        # 下载文件
        zip_name = f"batch_{timestamp}.zip"
        local_zip_path = os.path.join(DOWNLOAD_DIR, zip_name)
        cmd = f'/home/qyy/oss cp oss://csidatasets/{zip_name} {local_zip_path}'
        
        child = pexpect.spawn(cmd, encoding='utf-8', timeout=120)
        child.expect("Download successfully", timeout=180)
        print(f"[下载完成] {zip_name}")

        # 启动独立线程解压
        worker = UnzipWorker(local_zip_path)
        worker.start()
        return True

    except pexpect.EOF:
        print("[错误] 下载失败，文件可能不存在")
    except pexpect.TIMEOUT:
        print("[错误] 下载超时，检查文件大小或网络")
    finally:
        if 'child' in locals() and child.isalive():
            child.close()

def monitor_oss(interval=5):
    """监听OSS文件变化并触发下载解压"""
    print("启动OSS文件监听服务...")
    Path(EXTRACT_DIR).mkdir(parents=True, exist_ok=True)

    while True:
        try:
            current_latest = run_check('/home/qyy/oss ls -s -d oss://csidatasets/')
            recorded_latest = _read_latest_timestamp()

            if current_latest and (not recorded_latest or current_latest > recorded_latest):
                print(f"发现新文件 batch_{current_latest}.zip，开始处理...")
                if run_download(current_latest):
                    _write_latest_timestamp(current_latest)
                else:
                    print("下载失败，等待下次检查")

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[信息] 用户终止监听")
            break
        except Exception as e:
            print(f"[监听异常] {e}")
            time.sleep(interval)

def _read_latest_timestamp():
    """读取已记录的最新时间戳"""
    try:
        with open(TIMESTAMP_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def _write_latest_timestamp(timestamp):
    """写入最新时间戳"""
    with open(TIMESTAMP_FILE, 'w') as f:
        f.write(timestamp)

if __name__ == "__main__":
    monitor_oss(interval=5)