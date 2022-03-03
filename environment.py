from random import choice
from turtle import *
from freegames import floor, vector

from random import choice
from turtle import *
from base import vector,floor,square

## create a gym environment for pacman from scratch

class environment:
  def __init__(self):
    self.state = { 'score':0 }
    self.path = Turtle(visible = False)
    self.writer = Turtle(visible = False)

    self.aim = vector(5, 0)
    self.pacman = vector(-10, -40) # -40, -80
    self.done = False
    self.ghosts = [
        [vector(0, 0), vector(5,0)],
    ]
    #    [vector(-180, 160), vector(0,5)],
    #    [vector(100, 160), vector(0, -5)],
    #    [vector(100, -160), vector(-5, 0)]
    #    ]
    self.ghosts = []
    self.tilesinit = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    #    self.tilesinit = [
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #]
    
    #self.tilesinit = [
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    #    0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    #    0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #    0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
    #    0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #]
    self.tiles = self.tilesinit.copy()
    
  def square(self,x, y):
        self.path.up()
        self.path.goto(x,y)
        self.path.down()
        self.path.begin_fill()

        for _ in range(4):
            self.path.forward(20)
            self.path.left(90)

        self.path.end_fill()

  def offset(self, point):
        x = (floor(point.x, 20) + 200) / 20
        y = (180 - floor(point.y, 20)) / 20
        index = int(x + y * 20)
        return index

  def valid(self,point):
        index = self.offset(point)
        if self.tiles[index] == 0:
            return False
        index = self.offset(point + 19) #19
        if self.tiles[index] == 0:
            return False
        return ((point.x % 20 == 0) or (point.y % 20 == 0))

  def world(self):
        bgcolor('black')
        self.path.color('blue')

        for index in range(len(self.tiles)):
            tile = self.tiles[index]
            if tile > 0:
                x = (index % 20) * 20 - 200
                y = 180 - (index // 20) * 20
                self.square(x,y)
                if tile == 1:
                    self.path.up()
                    self.path.goto(x + 10, y + 10)
                    self.path.dot(2, 'white')
                    
  def move(self):
        # Movement for Pacman
        self.writer.undo()
        self.writer.write(self.state['score'])
        clear()
        if self.valid(self.pacman + self.aim):
            self.pacman.move(self.aim)
        else: self.state['score'] = -10

        index = self.offset(self.pacman)
        if self.tiles[index] == 1:
            self.tiles[index] = 2
            self.state['score'] = 10
            x = (index % 20) * 20 - 200
            y = 180 - (index // 20) * 20
            self.square(x, y)
        up()
        goto(self.pacman.x + 10, self.pacman.y + 10)
        dot(20, 'yellow')

        # Movement for ghosts

        for point, course in self.ghosts:
            if self.valid(point + course):
                point.move(course)
            else:
                options = [
                    vector(5,0), vector(-5,0),
                    vector(0,5), vector(0, -5)
                    ]
                plan = choice(options)
                course.x = plan.x
                course.y = plan.y

            up()
            goto(point.x + 10, point.y + 10)
            dot(20, 'red')

        #update()

        # checking for collisions.

        for point,course in self.ghosts:
            if abs(self.pacman - point) < 20:
                self.state['score'] = -100
                self.done = True
                return
        
        if 1 not in self.tiles:
          self.state['score'] = 100
          self.done = True
          return

        #ontimer(self.move, 100)

  def change(self,x, y):
        # changing pacman aim if valid
        if self.valid(self.pacman + vector(x,y)):
            self.aim.x = x
            self.aim.y = y
            
  def step(self, action):
    # Execute one time step within the environment
    self.state['score'] =  0
    if action == "l":
      self.change(-5,0)
    elif action == "r":
      self.change(5,0)
    elif action == "u":
      self.change(0,5)
    elif action == "d":
      self.change(0,-5)
    self.move()
    if self.state['score'] == 0:
      self.state['score'] =  - 1
    reward = self.state['score']
    if self.done == False: 
      update()
    else:
      reset()
    pos_pacman = self.offset(self.pacman)
    obs_space = self.tiles.copy()
    obs_space[pos_pacman] = 3 
    for point, _ in self.ghosts:
      pos_ghost = self.offset(point)
      obs_space[pos_ghost] = 4
    return tuple(obs_space), reward, self.done

  def reset(self):
    self.state['score'] = 0
    self.world()
    hideturtle()
    tracer(False)
    self.path = Turtle(visible = False)
    self.writer = Turtle(visible = False)
    self.writer.goto(160, 160)
    self.writer.color('white')
    self.writer.write(self.state['score'])
    self.aim = vector(5, 0)
    self.pacman = vector(-10, 40)
    self.done = False
    self.ghosts = [
        [vector(0, 40), vector(5,0)],
    ]
    #    [vector(-180, 160), vector(0,5)],
    #    [vector(100, 160), vector(0, -5)],
    #    [vector(100, -160), vector(-5, 0)]
    #    ]
    #self.ghosts = []
    self.tiles = self.tilesinit.copy()
    pos_pacman = self.offset(self.pacman)
    obs_space = self.tiles.copy()
    obs_space[pos_pacman] = 3 
    for point, _ in self.ghosts:
      pos_ghost = self.offset(point)
      obs_space[pos_ghost] = 4
    return tuple(obs_space)



if __name__ == '__main__':
  setup(420, 420, 370, 0)
  hideturtle()
  tracer(False)
  env = environment()
  env.reset()
  #listen()
  #onkey(lambda: env.change(5,0), 'Right')
  #onkey(lambda: env.change(-5,0), 'Left')
  #onkey(lambda: env.change(0,5), 'Up')
  #onkey(lambda: env.change(0,-5), 'Down')
  for i in range(300):
    obs_space, reward, done = env.step('l')
  print(len(obs_space))
  env.reset()
  for i in range(300):
    tiles, reward, done = env.step('r')
  print(reward)