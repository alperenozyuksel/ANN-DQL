import pygame

pygame.init()

WIDTH = 450
HEIGHT = 360


black = (0, 0, 0)
white = (255, 255, 255)


pencere = pygame.display.set_mode((WIDTH, HEIGHT))

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

pygame.draw.line(pencere, RED, (0,0), (150,250), 5)
pygame.draw.line(pencere, WHITE, (150,250), (260,350), 5)

durum = True
while durum:
    for etkinlik in pygame.event.get():
        print(etkinlik)
        if etkinlik.type == pygame.QUIT:
            durum = False

    pygame.display.update()



pygame.quit()




