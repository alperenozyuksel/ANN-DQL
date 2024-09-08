import pygame
import random

from fontTools.tfmLib import PASSTHROUGH

FPS = 60

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 15

BALL_WIDTH = 20
BALL_HEIGHT = 20

PADDLE_SPEED = 3
BALL_X_SPEED = 2
BALL_Y_SPEED = 2

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

def drawPaddle(switch, paddleYPos):

    if switch == "left":
        paddle = pygame.Rect(PADDLE_BUFFER, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    elif switch == "right":
        paddle = pygame.Rect(WINDOW_WIDTH-PADDLE_BUFFER-PADDLE_WIDTH, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)


    pygame.draw.rect(screen, WHITE, paddle)


def drawBall(ballXPos, ballYPos):

    ball = pygame.Rect(ballXPos,ballYPos,BALL_WIDTH,BALL_HEIGHT)

    pygame.draw.rect(screen, BLACK, ball)


def updatePaddle(switch, action, paddleYPos, ballYPos):
    dft = 7.5
    if switch == "left":
        if action == 1:
            paddleYPos = paddleYPos - dft*PADDLE_SPEED
        if action == 2:
            paddleYPos = paddleYPos + dft*PADDLE_SPEED

        if paddleYPos < 0:
            paddleYPos = 0

        if paddleYPos > WINDOW_HEIGHT - PADDLE_HEIGHT:
            paddleYPos = WINDOW_HEIGHT - PADDLE_HEIGHT


    elif switch == "right":
        if paddleYPos + PADDLE_HEIGHT/2 < ballYPos + BALL_HEIGHT/2:
            paddleYPos = paddleYPos + PADDLE_SPEED*dft

        if paddleYPos + PADDLE_HEIGHT/2 > ballYPos + BALL_HEIGHT/2:
            paddleYPos = paddleYPos - PADDLE_SPEED*dft

        if paddleYPos < 0:
            paddleYPos = 0

        if paddleYPos > WINDOW_HEIGHT - PADDLE_HEIGHT:
            paddleYPos = WINDOW_HEIGHT - PADDLE_HEIGHT

class PongGame:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Pong Game')

        self.paddle1YPos = WINDOW_HEIGHT/2 - PADDLE_HEIGHT/2
        self.paddle2YPos = WINDOW_HEIGHT/2 - PADDLE_HEIGHT/2



        self.ballXPos = WINDOW_WIDTH/2

        self.clock = pygame.time.Clock()

        self.GScore = 0.0

        self.ballXDirection = random.sample([-1,1],1)[0]
        self.ballYDirection = random.sample([-1,1],1)[0]

        self.ballYPos = random.randint(0,9)*(WINDOW_HEIGHT-BALL_HEIGHT)/9


    def InitialDisplay(self):

        pygame.event.pump()

        screen.fill(BLACK)

        drawPaddle("left", self.paddle1YPos)
        drawPaddle("right", self.paddle2YPos)


        drawBall(self.ballXPos, self.ballYPos)


        pygame.display.flip()



    def PlayNextMove(self,action):



        DeltaFrameTime = self.clock.tick(FPS)

        pygame.event.pump()

        score = 0

        screen.fill(BLACK)

        self.paddle1YPos = updatePaddle("left", action, self.paddle1YPos, self.ballYPos)
        drawPaddle("left", self.paddle1YPos)

        self.paddle2YPos = updatePaddle("right", action, self.paddle2YPos, self.ballYPos)
        drawPaddle("right", self.paddle2YPos)


pg = PongGame()
pg.InitialDisplay()






