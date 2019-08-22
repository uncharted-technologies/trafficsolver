import numpy as np
import environments.rendering as rendering
import copy

class TrafficSimSimple():

    def __init__(self):

        class action_space():
            def __init__(self,n_actions): 
                self.n = n_actions

        class observation_space():
            def __init__(self,n_features): 
                self.shape = [n_features]

        self.action_space = action_space(2)
        self.observation_space = observation_space(4)

        self.GRID_SHAPE = (5,7)
        self.TRAFFIC_INTENSITY = 0.3
        self.MAX_TIMESTEPS = 50

        self.current_timestep = 0

        self.red_car_positions = np.zeros(self.GRID_SHAPE[1])
        self.blue_car_positions = np.zeros(self.GRID_SHAPE[1])

        self.traffic_light_state = np.zeros(2)
        self.car_state = np.zeros(2)

        self.red_car_positions_last = copy.deepcopy(self.red_car_positions) #For rendering purposes only
        self.blue_car_positions_last = copy.deepcopy(self.blue_car_positions) #For rendering purposes only

        self.viewer = None

    def step(self,action):

        done = False
        self.current_timestep += 1
        reward = 0

        #Check if time is up
        if self.current_timestep >= self.MAX_TIMESTEPS:
            done = True

        #Change state of traffic light chosen by action
        self.traffic_light_state[action] = (1 - self.traffic_light_state[action])

        self.red_car_positions_last = copy.deepcopy(self.red_car_positions) #For rendering purposes only
        self.blue_car_positions_last = copy.deepcopy(self.blue_car_positions) #For rendering purposes only

        #Move red cars by one, starting with car furthest to the right
        for i in reversed(range(self.GRID_SHAPE[1])):

            #Remove last car from grid
            if i == self.GRID_SHAPE[1]-1:
                if self.red_car_positions[i] == 1:
                    self.red_car_positions[i] = 0
                    #Get a point for every car that leaves the intersection
                    reward = 1
            
            #If car is before the red light, don't move it
            elif i == (self.GRID_SHAPE[1]-3)/2 and self.traffic_light_state[0] == 1:
                pass

            #If no red car in front, then move car forward by 1
            else:
                if self.red_car_positions[i] == 1 and self.red_car_positions[i+1] == 0:
                    self.red_car_positions[i] = 0
                    self.red_car_positions[i+1] = 1

        #Move blue cars by one, starting with car furthest to the right
        for i in reversed(range(self.GRID_SHAPE[1])):

            #Remove last car from grid
            if i == self.GRID_SHAPE[1]-1:
                if self.blue_car_positions[i] == 1:
                    self.blue_car_positions[i] = 0
                    #Get a point for every car that leaves the intersection
                    reward = 1
            
            #If car is before the red light, don't move it
            elif i == (self.GRID_SHAPE[1]-3)/2 and self.traffic_light_state[1] == 1:
                pass

            #If no blue car in front, then move car forward by 1
            else:
                if self.blue_car_positions[i] == 1 and self.blue_car_positions[i+1] == 0:
                    self.blue_car_positions[i] = 0
                    self.blue_car_positions[i+1] = 1

        #Check for collision at intersection. If there is a collision, terminate episode
        if self.blue_car_positions[int((self.GRID_SHAPE[1]-1)/2)] == 1 and self.red_car_positions[int((self.GRID_SHAPE[1]-1)/2)] == 1:
            done = True

        #Randomly generate cars at the start
        if np.random.uniform() < self.TRAFFIC_INTENSITY and self.red_car_positions[0] == 0:
            self.red_car_positions[0] = 1
        if np.random.uniform() < self.TRAFFIC_INTENSITY and self.blue_car_positions[0] == 0:
            self.blue_car_positions[0] = 1

        #Is red car now in front of traffic light? Include in state description
        if self.red_car_positions[int((self.GRID_SHAPE[1]-3)/2)] == 1:
            self.car_state[0] = 1
        else:
            self.car_state[0] = 0

        #Is blue car now in front of traffic light? Include in state description
        if self.blue_car_positions[int((self.GRID_SHAPE[1]-3)/2)] == 1:
            self.car_state[1] = 1
        else:
            self.car_state[1] = 0

        state = np.concatenate((self.car_state,self.traffic_light_state))

        return state, reward, done, {}

    def reset(self):
        self.current_timestep = 0
        self.red_car_positions = np.zeros(self.GRID_SHAPE[1])
        self.blue_car_positions = np.zeros(self.GRID_SHAPE[1])

        self.red_car_positions_last = copy.deepcopy(self.red_car_positions) #For rendering purposes only
        self.blue_car_positions_last = copy.deepcopy(self.blue_car_positions) #For rendering purposes only
        
        self.traffic_light_state = np.zeros(2)
        self.traffic_light_old_state = np.zeros(2)
        self.car_state = np.zeros(2)

        state = np.concatenate((self.car_state,self.traffic_light_state))

        return state

    def seed(self,seed):
        return

    def render(self):

        screen_width = 600
        screen_height = 400

        carwidth = 30
        carheight = 30

        traffic_light_radius = 10

        l,r,t,b = -carwidth/2, carwidth/2, carheight/2, -carheight/2

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)


        for i in range(self.GRID_SHAPE[1]):

            if self.red_car_positions_last[i] == 1:

                car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                car.set_color(1,0,0)

                cartrans = rendering.Transform()
                car.add_attr(cartrans)
                cartrans.set_translation(20 + 80*i, 300)

                self.viewer.add_onetime(car)


        for i in range(self.GRID_SHAPE[1]):

            if self.blue_car_positions_last[i] == 1:

                if i < (self.GRID_SHAPE[1]-1)/2:

                    car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                    car.set_color(0,0,1)

                    cartrans = rendering.Transform()
                    car.add_attr(cartrans)
                    cartrans.set_translation(260, 20 + 100*i)

                    self.viewer.add_onetime(car)

                else:

                    car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                    car.set_color(0,0,1)

                    cartrans = rendering.Transform()
                    car.add_attr(cartrans)
                    cartrans.set_translation(20 + 80*i, 300)

                    self.viewer.add_onetime(car)
        
        light1 = rendering.make_circle(traffic_light_radius)
        if self.traffic_light_state[0] == 0:
            light1.set_color(0,1,0)
        else:
            light1.set_color(1,0,0)
        lighttrans = rendering.Transform()
        light1.add_attr(lighttrans)
        lighttrans.set_translation(220, 350)
        self.viewer.add_onetime(light1)

        light2 = rendering.make_circle(traffic_light_radius)
        if self.traffic_light_state[1] == 0:
            light2.set_color(0,1,0)
        else:
            light2.set_color(1,0,0)
        lighttrans = rendering.Transform()
        light2.add_attr(lighttrans)
        lighttrans.set_translation(300,250)
        self.viewer.add_onetime(light2)


        return self.viewer.render(return_rgb_array = False)
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None