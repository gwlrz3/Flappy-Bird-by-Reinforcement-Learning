import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature1 import BrainDQN
from UserInput import *
import numpy as np
import pygame
from pygame.locals import *
# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))

def playFlappyBird():
	# Step 1: init BrainDQN
	actions = 2
	brain = BrainDQN(actions)
	# Step 2: init Flappy Bird Game
	flappyBird = game.GameState()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = np.array([1,0])  # do nothing
	observation0, reward0, terminal = flappyBird.frame_step(action0)
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	brain.setInitState(observation0)


	# player can play for 5 times and then start the train process
	i = 0
	while True:
		action = np.array([1,0])
		for e in pygame.event.get():
			if e.type == MOUSEBUTTONUP or (e.type == KEYUP and e.key in (K_UP, K_RETURN, K_SPACE)):
				action = np.array([0,1])
				# i+=1
				# print("click",i)
				break
		nextObservation,reward,terminal = flappyBird.frame_step(action)
		nextObservation = preprocess(nextObservation)
		time = brain.trainPerception(nextObservation,action,reward,terminal)
		print(time)
		# if reward == -1:
		# 	i+=1
		# if i >= 5:
		# 	break
		if time >= 1000 and reward == -1:
			break

	# count the total score before the bird die 5 times.
	count = 0
	total = 0
	# Step 3.2: run the game
	while 1!= 0:
		action = brain.getAction()
		nextObservation,reward,terminal = flappyBird.frame_step(action)
		nextObservation = preprocess(nextObservation)
		brain.setPerception(nextObservation,action,reward,terminal,0)
		if reward >= 1:
			total += 1
		if reward == -1:
			count += 1
		print(count,total)
		# if count >= 5:
		# 	print(total)
			# break

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()
