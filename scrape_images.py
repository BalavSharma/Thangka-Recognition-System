import os
import time
import requests
from duckduckgo_search import DDGS
from urllib.parse import urlparse

# Extracted exact sub-folders from the Dataset directory to replicate
classes = {
    "Amitabha": "Amitabha Thangka traditional painting high resolution",
    "Avalokitesvara": "Avalokitesvara Chenrezig Thangka painting",
    "Bhaisajyaguru": "Bhaisajyaguru Medicine Buddha Thangka painting precise",
    "BuddhaShakyamuni": "Buddha Shakyamuni Thangka painting high details",
    "GreenTara": "Green Tara Thangka very detailed precise painting",
    "Mahakala": "Mahakala protector Thangka high resolution painting",
    "Maitreya": "Maitreya Future Buddha Thangka painting precise",
    "Manjushri": "Manjushri Bodhisattva Thangka sword wisdom painting",
    "Milerapa": "Milarepa Yogi Thangka Tibetan painting",
    "Padmasambhava": "Padmasambhava Guru Rinpoche Thangka authentic",
    "Samantabhadra": "Samantabhadra Thangka high resolution precise",
    "WhiteTara": "White Tara Thangka painting authentic",
    "Zambhala": "Dzambhala wealth deity Thangka painting",
    "Zhabdrung_Ngawang_Namgyel": "Zhabdrung Ngawang Namgyel Thangka Bhutan painting"
}

output_dir = "Web-scrapped Images"
os.makedirs(output_dir, exist_ok=True)

MAX_IMAGES_PER_CLASS = 100

def download_image(url, save_dir, filename):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        # Fast duplicate check using HEAD req
        head = requests.head(url, timeout=3, headers=headers, allow_redirects=True)
        if head.status_code == 200:
            content_length = head.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                content_size = int(content_length)
                for fname in os.listdir(save_dir):
                    if os.path.getsize(os.path.join(save_dir, fname)) == content_size:
                        return False # duplicate found without downloading payload
                        
        response = requests.get(url, timeout=5, headers=headers)
        response.raise_for_status()
        
        if "image" not in response.headers.get("Content-Type", ""): return False
            
        # 2nd safety check
        content_size = len(response.content)
        for fname in os.listdir(save_dir):
            if os.path.getsize(os.path.join(save_dir, fname)) == content_size: return False
                
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        return False

print("Starting precise image scraping...")

with DDGS() as ddgs:
    for cls_dir, query in classes.items():
        print(f"\n--- Scraping for {cls_dir} ---")
        save_dir = os.path.join(output_dir, cls_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        count = len(os.listdir(save_dir))
        if count >= MAX_IMAGES_PER_CLASS:
            print(f"Already have {count} images for {cls_dir}. Skipping...")
            continue
        try:
            # We fetch more results in case some links are broken
            results = ddgs.images(
                query,
                safesearch="off",
                size="Large",
                type_image="photo",
                max_results=MAX_IMAGES_PER_CLASS * 3 
            )
            
            if not results:
                print(f"No results found for {cls_dir}.")
                continue
                
            for r in results:
                if count >= MAX_IMAGES_PER_CLASS:
                    break
                    
                image_url = r['image']
                
                # Derive extension or use default
                ext = ".jpg"
                if image_url.lower().endswith(".png"): ext = ".png"
                elif image_url.lower().endswith(".webp"): ext = ".webp"
                elif image_url.lower().endswith(".jpeg"): ext = ".jpeg"
                
                filename = f"{cls_dir}_{count:03d}{ext}"
                
                if download_image(image_url, save_dir, filename):
                    count += 1
                    print(f"Downloaded [{count}/{MAX_IMAGES_PER_CLASS}] for {cls_dir}")
                
        except Exception as e:
            print(f"Error scraping {cls_dir}: {e}")

print("\n✔ Web-scraping completed successfully! Check the 'Web-scrapped Images' directory.")
