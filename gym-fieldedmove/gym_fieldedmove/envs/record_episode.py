import pickle

from gym_fieldedmove.envs.fieldedmove_env import FieldedMove
import pygame
import time
import numpy as np

successes, failures = pygame.init()
print("Initializing pygame: {0} successes and {1} failures.".format(successes, failures))


def main():
    env = FieldedMove()
    env.set_mode('human')
    env.reset()
    magnitude = float(input("Choose force magnitude : "))
    env.magnitude = magnitude
    period = float(input("Choose period of force : "))
    env.period_x = env.period_y = period
    goal_degree = float(input("Choose goal degree : "))
    goal_degree = (goal_degree/180) * np.pi
    goal = np.array([env.rad * np.sin(goal_degree), env.rad * np.cos(goal_degree)])
    env.goal = env.center + goal
    # env.reset()
    running = True
    target = np.zeros(2)
    f = 1
    value = 0
    t = 0
    record = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            target[1] = -f
            # elif event.type == pygame.KEYDOWN:
                # if event.key == pygame.K_w:
                #     target[1] -= f
                # elif event.key == pygame.K_s:
                #     target[1] += f
                # elif event.key == pygame.K_a:
                #     target[0] -= f
                # elif event.key == pygame.K_d:
                #     target[0] += f
                # target = [f,0]
            # elif event.type == pygame.KEYUP:
                # if event.key == pygame.K_w:
                #     target[1] += f
                # elif event.key == pygame.K_s:
                #     target[1] -= f
                # elif event.key == pygame.K_a:
                #     target[0] += f
                # elif event.key == pygame.K_d:
                #     target[0] -= f

        state, reward, done, info = env.step(target)
        record.append(state)
        value += reward
        t += 1
        if t > 50:
            print("cumulative reward : ", value)
            value = 0
            break
        time.sleep(0.1)

    record = np.array(record)
    filename = input("saving filename : ")
    with open(filename, 'wb') as f :
        pickle.dump(record, f)

    # plt.imshow(state)
    # plt.show()

    print("Exited the game loop. Game will quit...")
    quit()


if __name__ == "__main__":
    main()
