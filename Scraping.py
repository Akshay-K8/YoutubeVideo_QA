from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def extract_transcript_text(youtube_url: str) -> str:
    options = Options()
    options.add_argument('--headless')  # Optional
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--start-maximized')

    driver = webdriver.Chrome(options=options)

    try:
        print("🔗 Opening youtubetotranscript.com...")
        driver.get("https://youtubetotranscript.com/")

        print("🔍 Waiting for input field...")
        input_box = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "youtube_url"))
        )
        input_box.clear()
        input_box.send_keys(youtube_url)

        print("▶️ Clicking 'Get Free Transcript' button...")
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]'))
        )
        submit_button.click()

        print("⏳ Waiting for transcript div to load...")
        transcript_div = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.ID, "transcript"))
        )
        
        # Wait a little for text to fully render
        time.sleep(2)

        transcript_text = transcript_div.text.strip()

        if not transcript_text:
            raise Exception("Transcript is empty.")

        print("✅ Transcript extracted successfully.")
        return transcript_text

    except Exception as e:
        print(f"[Error] Transcript extraction failed: {e}")
        return None

    finally:
        driver.quit()
