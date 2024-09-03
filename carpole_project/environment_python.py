
import pygame
import random

widht = 360
height = 360

fps = 30

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# initiliaze pygame and create window

pygame.init()
screen = pygame.display.set_mode((widht, height))
pygame.display.set_caption('RL Game')
clock = pygame.time.Clock()

# game loop
running = True
while running:
    # keep loop running at the right speed
    clock.tick(fps)

    # procces input
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    # update

    # draw / render (show)
    screen.fill(white)
    # after drawing flip display

    pygame.display.flip()


pygame.quit()
