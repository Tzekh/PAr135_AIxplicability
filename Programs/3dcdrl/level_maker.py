from matplotlib import pyplot as plt
import matplotlib.patches as patches

class LineBuilder:
    def __init__(self, line, dbclick=False):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.connection = False
        self.last_point = [0, 0]
        self.dbclick = dbclick
        self.walls = []
        self.spawn = []

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return

        if event.button == 3 and self.dbclick :
            self.line.set_data(round_coordinates([event.xdata, event.ydata], distance=10 ))
            self.line.figure.canvas.draw()
            self.spawn = round_coordinates([event.xdata, event.ydata], distance=10 )
        elif not self.dbclick and event.button == 1:
            print("OK")
            if not self.connection:
                self.last_point = [event.xdata, event.ydata]
            else:
                x0, x1, y0, y1 = round_coordinates(make_it_straight(self.last_point[0], event.xdata, self.last_point[1], event.ydata), distance=50)
                self.xs.append([x0, x1])
                self.ys.append([y0, y1])
                self.walls.append([x0, y0, x1, y1])
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()

            self.connection = not self.connection

def draw_level(height = 2000, width = 2000):

    fig = plt.figure()

    axes = plt.gca()
    axes.set_xlim(-50, width+50)
    axes.set_ylim(-50, height+50)

    ax = fig.add_subplot(111)
    ax.set_title('click to add points')
    line, = ax.plot([], [], linestyle="-", marker=".", color="r")
    line2, = ax.plot([], [], linestyle="none", marker="o", color="g")
    rect = patches.Rectangle((0, 0), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    linebuilder = LineBuilder(line)
    linebuilder2 = LineBuilder(line2, dbclick=True)

    plt.show()

    return linebuilder.walls, linebuilder2.spawn

def round_coordinates(coordinates, distance = 10):
    new_coordinates = []
    for coordinate in coordinates:
        new_coordinates.append(coordinate//distance*distance+(distance if coordinate%distance>distance/2 else 0))
    return new_coordinates

def make_it_straight(x0, x1, y0, y1):

    if abs(x0-x1) > abs(y0-y1):
        y1=y0
        if x1<x0:
            x1, x0 = x0, x1
    else:
        x1 = x0
        if y1<y0:
            y0, y1 = y1, y0

    return x0, x1, y0, y1