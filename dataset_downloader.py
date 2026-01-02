from icrawler.builtin import BingImageCrawler
import os

def download(keyword, folder, limit=40):
    os.makedirs(folder, exist_ok=True)
    crawler = BingImageCrawler(storage={"root_dir": folder})
    crawler.crawl(
        keyword=keyword,
        max_num=limit,
        filters={"type": "photo"}
    )

print("Downloading healthy vetiver images...")
download("vetiver grass plant healthy", "dataset/healthy", 40)

print("Downloading moisture stressed vetiver images...")
download("vetiver grass dry leaves", "dataset/low_moisture", 40)

print("Downloading nutrient stressed vetiver images...")
download("vetiver grass yellow leaves", "dataset/low_nutrient", 40)

print("Download completed")

