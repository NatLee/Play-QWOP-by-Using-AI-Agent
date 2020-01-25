
import time

import numpy as np
import cv2

from pynput.keyboard import Key, Controller

from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


class BrowserController:

    def __init__(self, driver_path):

        # start a browser to enter QWOP
        self.driver = webdriver.Chrome(executable_path=driver_path)
        self.driver.get('http://www.foddy.net/Athletics.html')
        
        # wait for loading the game
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, 'window1')))
        time.sleep(3)
        
        # remove unnecessary components
        self.driver.execute_script("""
            var element = document.getElementById('window1');
            document.body.innerHTML = '';
            document.body.appendChild(element)
        """)

        # get width and height of the game
        self.canvas = self.driver.find_element_by_id('window1')
        self.canvas_width = self.canvas.size.get('width')
        self.canvas_height = self.canvas.size.get('height')
        
        # get the status of the game by checking color on the canvas
        self.check_point = (int(self.canvas_width * 0.225), int(self.canvas_height * 0.72))
        self.check_pixel_color = self.get_pixel(self.check_point)
        
        # get the location of score from the canvas
        self.score_location_x1 = int(self.canvas_width * 0.315)
        self.score_location_y1 = int(self.canvas_height * 0.05)
        self.score_location_x2 = int(self.canvas_width * 0.655)
        self.score_location_y2 = int(self.canvas_height * 0.13)
        
        self.keyboard = Controller()

    def get_pixel(self, point):
        """get color from a pixel"""
        img = self.screen_shot()
        return img[point[1], point[0], :]

    def screen_shot(self):
        """get a screen shot from web driver"""
        png = self.driver.get_screenshot_as_png()
        nparr = np.frombuffer(png, np.uint8)
        result = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = result[:self.canvas_height, :self.canvas_width, :]
        return result

    def start_game(self):
        """start a game by clicking the screen"""
        self.canvas.click()
        return

    def reloading(self):
        """press r to reload the game"""
        self.keyboard.press('r')
        time.sleep(0.08)
        self.keyboard.release('r')
        return

    def random_go(self):
        """randomly press"""
        choice = np.random.choice(['q', 'w', 'o', 'p'])
        self.keyboard.press(choice)
        time.sleep(0.08)
        self.keyboard.release(choice)
        return

    def n(self):
        """do nothing"""
        return

    def q(self):
        """press q"""
        self.keyboard.press('q')
        time.sleep(0.08)
        self.keyboard.release('q')
        return

    def w(self):
        """press w"""
        self.keyboard.press('w')
        time.sleep(0.08)
        self.keyboard.release('w')
        return

    def o(self):
        """press o"""
        self.keyboard.press('o')
        time.sleep(0.08)
        self.keyboard.release('o')
        return

    def p(self):
        """press p"""
        self.keyboard.press('p')
        time.sleep(0.08)
        self.keyboard.release('p')
        return
