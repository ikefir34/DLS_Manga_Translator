import shutil
from roboflow import Roboflow
from paddleocr import PaddleOCR
from pathlib import Path
import cv2 
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil
import requests
import json

rf = Roboflow(api_key="TqGTmVEv1qL7EOjhFfIK")
project = rf.workspace().project("text_detector-lwgey")
roboflow_model = project.version(4).model

ocr_model = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

ARLIAI_API_KEY = "4d071f03-6cd6-4a7c-8fd5-ed9dc721424a" 
url = "https://api.arliai.com/v1/chat/completions"

#----------------------------------------------------------

def delete_yolo_annotation(file_path, target_index):
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            index = int(parts[0].strip())
            if index != target_index:
                lines.append(line)
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)

def delete_paddleocr_annotation(parent_dir, index_to_delete):
    parent_dir = Path(parent_dir)

    folders = [p for p in parent_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    folders_sorted = sorted(folders, key=lambda p: int(p.name))

    folder_to_delete = parent_dir / str(index_to_delete)
    shutil.rmtree(folder_to_delete)

    folders = [p for p in parent_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    folders_sorted = sorted(folders, key=lambda p: int(p.name))

#----------------------------------------------------------

def yolo_prediction(img_path, img_name):
    prediction = roboflow_model.predict(img_path, confidence=40, overlap=30, hosted=False)
    result = prediction.json()
    yolo_storage_path = "storage/yolo_annotation/"
    img_annotation_path=yolo_storage_path + img_name + ".txt"
    with open(img_annotation_path, 'w') as f:
        for index, i in enumerate(result['predictions']):
            f.write(f"{index}, {i['x']-i['width']/2}, {i['y']-i['height']/2}, {i['x']+i['width']/2}, {i['y']+i['height']/2}\n")

def read_yolo_annotation(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            index = int(parts[0].strip())
            x_min = float(parts[1].strip())
            y_min = float(parts[2].strip())
            x_max = float(parts[3].strip())
            y_max = float(parts[4].strip())
            boxes.append({
                "index": index,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })
    return boxes

#----------------------------------------------------------

def cropImage(input_path, coords):
    image = Image.open(input_path)
    cropped_image = image.crop(coords)
    cropped_image.save("storage\cropped_image.jpg")

def translate_text(text, target_lang="русский"):
    payload = json.dumps({
      "model": "Gemma-3-27B-it", 
      "messages": [
        {
            "role": "system", 
            "content": f"You are a professional translator. Translate to {target_lang}. Output ONLY the translation."
        },
        {"role": "user", "content": text}
      ],
      "temperature": 0.3,
      "max_completion_tokens": 1024
    })
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f"Bearer {ARLIAI_API_KEY}"
    }
    response = requests.post(url, headers=headers, data=payload)
    return response.json()['choices'][0]['message']['content']

def paddleocr_prediction(img_path, img_name, language = True):
    yolo_storage_path = "storage/yolo_annotation/"
    yolo_img_annotation_path=yolo_storage_path + img_name + ".txt"
    output_path = "storage/paddleocr_annotation/"+img_name
    Path(output_path).mkdir(parents=True, exist_ok=True)
    boxes = read_yolo_annotation(yolo_img_annotation_path)
    for box in boxes:
        coords = (box['x_min'], box['y_min'], box['x_max'], box['y_max'])
        cropImage(img_path, coords)
        result = ocr_model.predict(input="storage\cropped_image.jpg")
        output_path_annotation = output_path+"/"+str(box["index"])
        res = result[0]
        res.save_to_json(output_path_annotation)
        if language:
            original_text = ""
            for text in reversed(res["rec_texts"]):
                original_text+=text
            trans_text = translate_text(original_text)
        else:
            original_text = ""
            for text in res["rec_texts"]:
                original_text+=text
            trans_text = translate_text(original_text)
        text_file_path = Path(output_path_annotation) / "text.txt" 
        with open(text_file_path, "w", encoding="utf-8") as f: 
            f.write(trans_text)

def read_text_prediction(file_path):
    with open(file_path, "r", encoding="utf-8") as f: 
        return f.read()
    
def write_text_prediction(file_path, text):
    with open(file_path, "w", encoding="utf-8") as f: 
            f.write(text)

#----------------------------------------------------------

def download_directory(path, language):
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    yolo_storage_path = Path("storage/yolo_annotation")
    shutil.rmtree(yolo_storage_path)
    yolo_storage_path.mkdir(parents=True, exist_ok=True) 

    paddleocr_storage_path = Path("storage\paddleocr_annotation")
    shutil.rmtree(paddleocr_storage_path)
    paddleocr_storage_path.mkdir(parents=True, exist_ok=True) 
    
    storage_path = Path("storage/images")
    shutil.rmtree(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True) 
    directory_path = Path(path)
    
    for img_file in directory_path.rglob("*"):
        if img_file.is_file() and img_file.suffix.lower() in extensions:
            destination = storage_path / img_file.name
            img = Image.open(img_file)
            img.thumbnail((1024, 1024))
            img.save(destination, quality=85)
            yolo_prediction(str(destination), img_file.stem)
            paddleocr_prediction(str(destination), img_file.stem, language)

def download_file(path, language):
    yolo_storage_path = Path("storage/yolo_annotation")
    shutil.rmtree(yolo_storage_path)
    yolo_storage_path.mkdir(parents=True, exist_ok=True) 

    paddleocr_storage_path = Path("storage\paddleocr_annotation")
    shutil.rmtree(paddleocr_storage_path)
    paddleocr_storage_path.mkdir(parents=True, exist_ok=True) 

    storage_path = Path("storage/images")
    shutil.rmtree(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True) 
    image_path = Path(path)

    destination = storage_path / image_path.name
    img = Image.open(image_path)
    img.thumbnail((1024, 1024))
    img.save(destination, quality=85)  
    yolo_prediction(str(destination), image_path.stem)
    paddleocr_prediction(str(destination), image_path.stem, language)

#----------------------------------------------------------

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def draw_text_with_outline(draw, x, y, text, font,
                           fill=(0, 0, 0),
                           outline=(255, 255, 255),
                           outline_width=2):

    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline)

    draw.text((x, y), text, font=font, fill=fill)

def fit_text_to_box(draw, text, font_path, box_width, box_height,
                    max_font_size=40,
                    min_font_size=6,
                    adaptive_min_font=14,
                    line_spacing=1.3):
    words = text.split()

    font_size = max_font_size
    while font_size >= adaptive_min_font:
        font = ImageFont.truetype(font_path, font_size)

        too_wide = any(
            draw.textbbox((0, 0), w, font=font)[2] > box_width
            for w in words
        )

        if not too_wide:
            break

        font_size -= 1

    if font_size < adaptive_min_font:
        font_size = adaptive_min_font

    while font_size >= adaptive_min_font:
        font = ImageFont.truetype(font_path, font_size)

        lines = []
        current = ""

        for w in words:
            test = current + (" " if current else "") + w
            if draw.textbbox((0, 0), test, font=font)[2] <= box_width:
                current = test
            else:
                lines.append(current)
                current = w

        if current:
            lines.append(current)

        total_h = sum(
            int((draw.textbbox((0, 0), line, font=font)[3] -
                 draw.textbbox((0, 0), line, font=font)[1]) * line_spacing)
            for line in lines
        )

        if total_h <= box_height:
            return font, lines, line_spacing

        font_size -= 1

    font = ImageFont.truetype(font_path, adaptive_min_font)

    chars = list(text)
    lines = []
    buf = ""

    for ch in chars:
        test = buf + ch
        if draw.textbbox((0, 0), test, font=font)[2] <= box_width:
            buf = test
        else:
            lines.append(buf)
            buf = ch

    if buf:
        lines.append(buf)

    return font, lines, line_spacing

def apply_paddleocr_annotations(
        img_path,
        img_name,
        save_path,
        adaptive_min_font=14,
        text_color_hex="#000000",
        font_path="arial.ttf"
):
    img_path = str(Path(img_path).resolve()).replace("\\", "/")
    img = cv2.imread(img_path)

    if img is None:
        print("Ошибка: не удалось открыть изображение:", img_path)
        return

    new_h, new_w = img.shape[:2]

    original_path = Path("storage/images") / (img_name + Path(img_path).suffix)
    if not original_path.exists():
        original_path = Path(img_path)

    orig_w, orig_h = Image.open(original_path).size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    yolo_boxes = read_yolo_annotation(f"storage/yolo_annotation/{img_name}.txt")

    annotation_root = Path(f"storage/paddleocr_annotation/{img_name}")
    if not annotation_root.exists():
        print("Нет PaddleOCR разметки:", annotation_root)
        return

    text_rgb = hex_to_rgb(text_color_hex)

    for folder in sorted(annotation_root.iterdir(), key=lambda p: int(p.name)):
        idx = int(folder.name)

        det_json = folder / "cropped_image_res.json"
        text_file = folder / "text.txt"

        if not det_json.exists() or not text_file.exists():
            continue

        with open(det_json, "r", encoding="utf-8") as f:
            det = json.load(f)

        if not det.get("rec_boxes"):
            continue

        yolo_box = next((b for b in yolo_boxes if b["index"] == idx), None)
        if yolo_box is None:
            continue

        translated_text = read_text_prediction(text_file)

        all_gx0, all_gy0, all_gx1, all_gy1 = [], [], [], []

        for x0, y0, x1, y1 in det["rec_boxes"]:
            crop_x_min = yolo_box["x_min"] * scale_x
            crop_y_min = yolo_box["y_min"] * scale_y

            gx0 = int(crop_x_min + x0 * scale_x)
            gy0 = int(crop_y_min + y0 * scale_y)
            gx1 = int(crop_x_min + x1 * scale_x)
            gy1 = int(crop_y_min + y1 * scale_y)

            all_gx0.append(gx0)
            all_gy0.append(gy0)
            all_gx1.append(gx1)
            all_gy1.append(gy1)

        gx0 = min(all_gx0)
        gy0 = min(all_gy0)
        gx1 = max(all_gx1)
        gy1 = max(all_gy1)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts = np.array([[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        box_width = gx1 - gx0
        box_height = gy1 - gy0

        font, lines, spacing = fit_text_to_box(
            draw,
            translated_text,
            font_path,
            box_width,
            box_height,
            adaptive_min_font=adaptive_min_font
        )

        line_heights = []
        for line in lines:
            tb = draw.textbbox((0, 0), line, font=font)
            line_heights.append(tb[3] - tb[1])

        total_text_height = sum(int(h * spacing) for h in line_heights)
        start_y = gy0 + (box_height - total_text_height) // 2

        y = start_y
        for line in lines:
            tb = draw.textbbox((0, 0), line, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]

            x = gx0 + (box_width - text_w) // 2

            draw_text_with_outline(
                draw,
                x,
                y,
                line,
                font,
                fill=text_rgb,
                outline=(255, 255, 255)
            )

            y += int(text_h * spacing)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    output_path = f"{save_path}/{img_name}_translated.jpg"
    cv2.imwrite(output_path, img)


#----------------------------------------------------------

def get_image_number (num):
    storage_path = Path("storage/images")
    for i, img_file in enumerate(storage_path.rglob("*")):
        if i == (num-1):
            return img_file.name
        
def get_image_name (num):
    storage_path = Path("storage/images")
    for i, img_file in enumerate(storage_path.rglob("*")):
        if i == (num-1):
            return img_file.stem
        
def get_len_directory():
    storage_path = Path("storage/images")
    return len(list(storage_path.rglob("*")))