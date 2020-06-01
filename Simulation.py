import numpy as np
import matplotlib as mpl
import random
import operator
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm.autonotebook import tqdm


def world_dimensions(m, n):
    """
    creates a numpy array of m x n dimensions filled with zeros
    """
    return np.zeros((m, n), dtype=int)


def empty_positions(world):
    """
    returns a list with the coordinates of all the empty positions (zeros)
    """
    return list(zip(*np.where(world == 0)))


def healthy_cells(world):
    """
    returns a list with the coordinates of all the healthy cells (ones)
    """
    return list(zip(*np.where(world == 1)))


def infected_cells(world):
    """
    returns a list with the coordinates of all the infected cells (twos)
    """
    return list(zip(*np.where(world == 2)))


def immune_cells(world):
    """
    returns a list with the coordinates of all the immune cells (threes)
    """
    return list(zip(*np.where(world == 3)))


def dead_cells(world):
    """
    returns a list with the coordinates of all the dead cells (fours)
    """
    return list(zip(*np.where(world == 4)))


def randomize_cells(world, hl, inf, imm):
    """
    sets a random position for all healthy, infected, and immune cells given
    (i.e., the array is randomly filled with os, 1s, 2s and 3s)
    """
    healthy = [1 for x in range(hl)]
    infected = [2 for x in range(inf)]
    immune = [3 for x in range(imm)]

    for h in healthy:
        empty = empty_positions(world)
        pos = np.random.choice(len(empty) - 1, size=None, replace=True)
        pos_healthy = empty[pos]
        world[pos_healthy] = 1

    for i in infected:
        empty = empty_positions(world)
        pos = np.random.choice(len(empty) - 1, size=None, replace=True)
        pos_inf = empty[pos]
        world[pos_inf] = 2

    for im in immune:
        empty = empty_positions(world)
        pos = np.random.choice(len(empty) - 1, size=None, replace=True)
        pos_imm = empty[pos]
        world[pos_imm] = 3

    return world


def move_healthy(world, tuples):
    healthy = healthy_cells(world)
    for h in healthy:
        movement(world, h, tuples)


def move_infected(world, tuples, frames):
    infected = infected_cells(world)
    for inf in infected:
        infect(world, inf, tuples, frames)
        movement(world, inf, tuples)


def move_immune(world, tuples):
    immune = immune_cells(world)
    for imm in immune:
        movement(world, imm, tuples)


def infect(world, coord, tuples, frames):
    """
    if there is any healthy cell in an adjacent cell of an infected cell,
    then it is infected (from value 1 to 2)
    """
    poss = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (1, -1), (-1, 1), (-1, -1)]
    healthy = healthy_cells(world)

    for p in poss:
        x = tuple(map(operator.add, p, coord))
        if x in healthy:
            chance = np.random.choice([1, 2], size=None, replace=True, p=[0.9, 0.1])  # sets probability of infection:
                                                                                      # 1 = healthy, 2 = infected;
                                                                                      # p[no infection, infection]
            world[x] = chance
            if chance == 2:
                tuples.append([x, frames])


def movement(world, coord, tuples):
    """
    given a cell, it is moved to a random adjacent cell
    """
    empty = empty_positions(world)
    poss = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (1, -1), (-1, 1), (-1, -1)]
    free = []
    value_coord = world[coord]

    for p in poss:
        x = tuple(map(operator.add, p, coord))
        if x in empty:
            if world[x] == 0:
                free.append(x)

    if len(free) == 0:
        world[coord] = value_coord

    else:
        new_poss = random.choice(free)
        world[coord] = 0
        world[new_poss] = value_coord

        if value_coord == 2:
            flat_list(tuples, coord, new_poss)


def flat_list(tuples, coord, new_poss):
    """
    converts a list of lists to a simple list
    """
    j = 0
    for sublist in tuples:
        for item in sublist:
            if item == coord:
                tuples[int(j / 2)][0] = new_poss
            j += 1


def recover(world, tuples, frames):
    """
    for each infected cell, the frame in which it was infected is taken and it is subtracted from the actual frame.
    If the difference is greater or equal than some specific value, then the cell change its state to either immune
    or dead
    """
    for t in tuples:
        if frames - t[1] >= 20:
            ind = t[0]
            chance = np.random.choice([3, 4], size=None, replace=True,
                                      p=[0.9, 0.1])  # set the mortality: 3 = immune, 4 = dead; p[immune, dead]
            world[ind] = chance
            tuples.remove(t)


def list_of_tuples(world):
    """
    creates a list of tuples in which the first element of each pair is the coordinate of an infected cell,
    and the second the initial frame (0): [(x,y), 0]
    """
    infected = infected_cells(world)
    l_tuples = []
    for inf in infected:
        l_tuples.append([inf, 0])

    return l_tuples


def update_world(world, tuples, frames):
    """
    returns an updated state of the world
    """
    recover(world, tuples, frames)
    move_healthy(world, tuples)
    move_infected(world, tuples, frames)
    move_immune(world, tuples)

    return world


# Pre-processing: every operations is computed in order to run the simulation smoothly and avoid stuttering
healthy = 200
infected = 2
immune = 1
my_world = world_dimensions(30, 30) # creates a world with m x n dimensions
my_world = randomize_cells(my_world, healthy, infected, immune)  # number of healthy, infected, and immune cells
                                                            # randomly distributed in the world
tuples = list_of_tuples(my_world)  # creates a list of tuples (infected, frame) setting the initial frame to 0
population = len(list(zip(*(np.where((my_world > 0) & (my_world <= 4)))))) # calculate the whole population of cells

f = 100 # number of frames
inter = 300 # milliseconds between frames
p = []
for i in tqdm(range(f)):
    u = update_world(my_world, tuples, i)
    z = u.tolist()
    p.append(z)

pre_processed_world = p


# Executing simulation
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
colors = ['floralwhite', 'lightgreen', 'tomato', 'cornflowerblue', 'darkslategrey']
bounds = [0, 1, 2, 3, 4]
cmap = mpl.colors.ListedColormap(colors)
im = ax1.imshow(my_world, interpolation='none', cmap=cmap)
axtext = fig.add_axes([0.0, 0.95, 0.1, 0.05])  # add time at the top left corner of the figure
axtext.axis("off")
time = axtext.text(0.5, 0.5, str(0), ha="left", va="top")

def animate(i):
    im.set_array(pre_processed_world[i])
    time.set_text(str(i))

    return im, time,

anim = animation.FuncAnimation(fig, animate, frames=f,
                               interval=inter, blit=True, repeat=False)

plt.show()


# Some statistics
p = population
total_infected = len(list(zip(*(np.where((my_world > 1) & (my_world <= 4))))))  # number of infected
                                                                                # cells (infected + immune + dead)
total_dead = len(dead_cells(my_world)) # number of dead cells
prob_infected = total_infected / p * 100
mortality = total_dead / total_infected * 100

x = [1, 2]
y = [prob_infected, mortality]
width = 0.8
fig, ax = plt.subplots()
rects1 = ax.bar(x, y, width, color='cornflowerblue')
ax.set_ylim(0, 100)
ax.set_ylabel('Percentage')
ax.set_title('Percentage of infected cells and mortality')
plt.yticks(np.arange(0, 110, 10))
ax.set_xticks(np.add(x, (0)))
ax.set_xticklabels(('Infected population', 'Mortality',))

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height,
                '%d%%' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
plt.show()
