import re
import time

import cv2
import pytesseract
import numpy as np

from .BrowserController import BrowserController

class QWOP():

    def __init__(self, observation_size=(200, 125), driver_path='./webdriver/chromedriver'):
        # start a browser
        self.bc = BrowserController(driver_path=driver_path)
        # initial steps of game
        self.game_steps = 0
        # observation size
        self.observation_size = observation_size

    def start(self):
        """start a game and initial the steps"""
        self.bc.start_game()
        self.game_steps = 0
        return

    def reset(self):
        """reload to end the game and return a observation"""
        self.game_steps = 0
        self.bc.reloading()
        obs = cv2.resize(self.get_screen_shot(), self.observation_size) / 255.
        return obs

    def execute_action(self, action):
        """execute one of the actions QWOP"""
        self.bc.start_game() # click the screen to avoid the man auto fall down
        self.game_steps += 1
        # execute the action
        getattr(self.bc, action)()
        shot = self.get_screen_shot()
        # get distance score
        distance_score = self.get_score()
        # time bouns
        time_score = -(self.game_steps/(abs(distance_score)+1e5))
        score = distance_score + time_score
        # check whether the game is over or not
        done = self.is_done(shot)
        if done:
            self.reset()
        return cv2.resize(shot, self.observation_size) / 255., score, done

    def is_done(self, shot):
        """Check the status of the game by using the color of pixel"""
        pixel_color = self.bc.check_pixel_color
        point = self.bc.check_point
        if not np.array_equal(shot[point[1], point[0], :], pixel_color):
            return False
        return True

    def get_score(self):
        """get score from a screen shot by OCR"""
        img = self.get_screen_shot()[self.bc.score_location_y1:self.bc.score_location_y2, self.bc.score_location_x1:self.bc.score_location_x2, :]
        # convert to black and white
        img = ~(np.array(img)[:, :, 0])
        thres_img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]
        # get score by OCR
        score = pytesseract.image_to_string(image=thres_img)
        digits_rgx = re.compile("-?[0-9]+.?[0-9]")
        result = digits_rgx.findall(score)
        if len(result) > 0:
            score = result[0]
        else:
            score = '0'
        try:
            score = score.replace(',', '.') # aviod `,`
            reward = float(score)
        except Exception as e:
            print(e)
            reward = 0.
        return reward

    def get_screen_shot(self):
        """get a screen shot a.k.a observation"""
        return self.bc.screen_shot()

    def map_action(self, action):
        """map action to browser"""
        return (['q', 'w', 'o', 'p', 'n'])[action]