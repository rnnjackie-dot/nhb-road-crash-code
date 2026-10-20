import os
import base64
import json
import time
import subprocess
import urllib.request

UMI_EXE = os.environ.get("UMI_EXE", "Umi-OCR.exe")
DATA_ROOT = os.environ.get("DATA_ROOT", "raw_images")
API_URL = "http://127.0.0.1:1224/api/ocr"
OUTPUT_DIR = os.environ.get("OCR_OUTPUT_DIR", "ocr_results")


def is_server_ready():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        opener.open("http://127.0.0.1:1224/api/ocr/get_options", timeout=2)
        return True
    except Exception:
        return False


def start_umi_ocr():
    if is_server_ready():
        print("Umi-OCR 已在运行")
        return
    print("正在启动 Umi-OCR...")
    subprocess.Popen([UMI_EXE])
    for i in range(60):
        time.sleep(1)
        if is_server_ready():
            print(f"Umi-OCR 启动成功 ({i+1}s)")
            return
    raise RuntimeError("Umi-OCR 启动超时，请手动启动后重试")


def ocr_image(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = json.dumps({
        "base64": b64,
        "options": {
            "ocr.language": "简体中文",
            "tbpu.parser": "single_para",
            "data.format": "text",
        }
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    resp = opener.open(req, timeout=30)
    result = json.loads(resp.read().decode("utf-8"))
    if result.get("code") == 100:
        return result["data"]
    else:
        return f"[OCR失败: code={result.get('code')}, msg={result.get('data')}]"


def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    folders = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ])
    total_folders = len(folders)
    for idx, folder in enumerate(folders, 1):
        folder_path = os.path.join(DATA_ROOT, folder)
        images = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        out_file = os.path.join(OUTPUT_DIR, f"{folder}.txt")
        print(f"[{idx}/{total_folders}] 处理 {folder}，共 {len(images)} 张图片")
        with open(out_file, "w", encoding="utf-8") as fout:
            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                fout.write(f"=== {img_name} ===\n")
                try:
                    text = ocr_image(img_path)
                    fout.write(text + "\n\n")
                    print(f"  v {img_name}")
                except Exception as e:
                    fout.write(f"[错误: {e}]\n\n")
                    print(f"  x {img_name}: {e}")
        print(f"  -> 已保存到 {out_file}")


if __name__ == "__main__":
    start_umi_ocr()
    process_all()
    print("\n全部完成！")
