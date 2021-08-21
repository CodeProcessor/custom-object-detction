"""
Copyright (C) Zone24x7, Inc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by dulanj <dulanj@zone24x7.com>, 21 August 2021
"""
import glob

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


class CaptureScreenShorts:
    def __init__(self):
        options = Options()

        self.driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
        self.driver.set_page_load_timeout(8)
        self.driver.set_window_size(1000, 1000)

    def get_screen_shot(self, url, save_path):
        ret = False
        try:
            self.driver.get(url)
            screenshot = self.driver.save_screenshot(save_path)
            ret = True
        except TimeoutException:
            print("Timeout - skipping")
        return ret

    def cleanup(self):
        self.driver.quit()


def convert_to_jpg():
    import cv2
    for _file in glob.glob("screenshots/*.png"):
        print(_file)
        path, ext = _file.split('.')
        img = cv2.imread(_file)
        cv2.imwrite(path + '.jpg', img)


if __name__ == '__main__':
    convert_to_jpg()

    file_path = "structure-detection-data-set.txt"
    no_of_images = 20

    lines = None
    with open(file_path, 'r') as fp:
        lines = fp.readlines()

    screen_capture = CaptureScreenShorts()
    count = 1
    for i, line in enumerate(lines):
        print(line)
        ret = screen_capture.get_screen_shot(line, f'screenshots/{count}.jpg')
        if ret:
            count += 1
        if count > no_of_images:
            break

    screen_capture.cleanup()
