import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from artificial_neural_network.neural_net import Network


class FormattedButton(tk.Button):
    """
    A button without the default bevelled edge
    """

    def __init__(self, *args, **kwargs):
        tk.Button.__init__(self, *args, **kwargs)
        self.config(relief='flat', activebackground=None, overrelief='flat', borderwidth=0)


class UserCanvas(tk.Frame):
    """Allows the user to draw a number which can then be saved to file"""

    def __init__(self, master):
        super(UserCanvas, self).__init__(master, width=280, height=280, highlightthickness=0)
        self.configure(background=self.winfo_toplevel()['background'],
                       highlightbackground=self.winfo_toplevel()['background'])
        self.canvas = tk.Canvas(self, width=280, height=280, bg='black', highlightthickness=0, cursor='spraycan')
        clear_button = FormattedButton(self, text="Clear", command=self.clear_canvas)
        save_button = FormattedButton(self, text="Save", command=self.save)

        self.winfo_toplevel().title('User Canvas')
        self.canvas.bind('<ButtonPress-1>', self.set_line_start)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.start_x, self.start_y = None, None

        self.canvas.grid(row=0, columnspan=2, sticky='nsew')
        clear_button.grid(row=1, column=0, sticky='nsew')
        save_button.grid(row=1, column=1, sticky='nsew')

        self.grid_rowconfigure(0, weight=1, minsize=280)
        self.grid_columnconfigure(0, weight=1, minsize=140)
        self.grid_columnconfigure(1, weight=1, minsize=140)

        self.image = Image.new('L', (280, 280), 0)
        self.image_draw = ImageDraw.Draw(self.image)

    def clear_canvas(self):
        self.canvas.delete('line')
        self.image.paste(0, (0, 0, 280, 280))

    def get_image(self):
        return self.image.resize((28, 28))

    def get_image_array(self):
        return np.asarray(self.get_image()).flatten()

    def save(self):
        image_array = self.get_image_array()
        with open('data_sets/user_entries.csv', 'ab') as entry_file:
            np.savetxt(entry_file, image_array, newline=',', fmt='%u', header='\n0', comments='')

    def set_line_start(self, event):
        self.start_x, self.start_y = event.x, event.y

    def draw(self, event):
        self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, width=20,
                                tags='line', capstyle='round', joinstyle='round', fill='white')
        self.image_draw.line([(self.start_x, self.start_y), (event.x, event.y)], width=20, joint='curve', fill='white')
        self.set_line_start(event)


class NetworkGui(tk.Canvas):
    """Visualise the network by shading the nodes different colours depending on activation
    zoom and scroll by pressing the + and - keys respectively
    pan by dragging on the screen"""

    def __init__(self, master, network: Network, visible_nodes_per_layer=None):
        super(NetworkGui, self).__init__(master, width=600, height=600)
        self.winfo_toplevel().title('Neural Network')
        self.configure(background=self.winfo_toplevel()['background'],
                       highlightbackground=self.winfo_toplevel()['background'])

        zoom_ratio = 0.9
        self.bind_all('<minus>', lambda event: self.scale(tk.ALL, 300, 200, zoom_ratio, zoom_ratio))
        self.bind_all('<plus>', lambda event: self.scale(tk.ALL, 300, 200, 1 / zoom_ratio, 1 / zoom_ratio))
        self.bind('<ButtonPress-1>', self.scroll_start)
        self.bind('<B1-Motion>', self.scroll_move)

        self.network = network

        self.current_visible_nodes = np.zeros(len(self.network.nodes_per_layer), dtype=int)
        self.visible_nodes_row_labels = []
        self.nodes = []
        self.nodes_text = []
        self.prediction_label = self.create_text(450, self.winfo_reqheight() // 2 + 20)

        if visible_nodes_per_layer is None:
            self.visible_nodes_per_layer = self.network.nodes_per_layer
        else:
            self.visible_nodes_per_layer = visible_nodes_per_layer
        self.draw_network()

    def scroll_start(self, event):
        self.scan_mark(event.x, event.y)

    def scroll_move(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def display_activations(self, input_data: np.array):
        activations_stacked = self.network.query(input_data)
        activations = np.concatenate([activation_layer.flatten() for activation_layer in activations_stacked])
        node_colors = 255 * (activations - min(activations)) / (max(activations) - min(activations))
        for i, (activation_color, node, text) in enumerate(zip(node_colors, self.nodes, self.nodes_text)):
            self.itemconfig(node, fill='#' + 3 * '{0:02x}'.format(int(activation_color)), outline='black', width=0)
            self.itemconfig(text, fill='white' if activation_color < 128 else 'black')
        self.itemconfig(self.nodes[- (len(activations_stacked[-1]) - np.argmax(activations_stacked[-1]))],
                        outline='tan4', width=5)

    def change_visible_nodes_row(self, layer, adjustment):
        self.itemconfigure(f'layer{layer}node_row{self.current_visible_nodes[layer]}', state='hidden')
        self.current_visible_nodes[layer] = (self.current_visible_nodes[layer] + adjustment) % (
                self.network.nodes_per_layer[layer] // self.visible_nodes_per_layer[layer])
        self.itemconfigure(f'layer{layer}node_row{self.current_visible_nodes[layer]}', state='normal')
        self.update_visible_node_layers()

    def update_visible_node_layers(self):
        for current_visible_node, label, total_nodes, visible_nodes in zip(self.current_visible_nodes,
                                                                           self.visible_nodes_row_labels,
                                                                           self.network.nodes_per_layer,
                                                                           self.visible_nodes_per_layer):
            number_of_layers = total_nodes // visible_nodes
            if number_of_layers > 1:
                self.itemconfig(label, text=f'{current_visible_node + 1}/' + f'{total_nodes // visible_nodes}')

    def get_node_y_coordinates(self, layer):
        visible_coordinates = np.linspace(70, self.winfo_reqheight() - 30,
                                          self.visible_nodes_per_layer[layer] + 1,
                                          endpoint=False)[1:]
        return np.tile(visible_coordinates, self.network.nodes_per_layer[layer] // self.visible_nodes_per_layer[layer])

    def draw_network(self, print_neurons=False):
        x_coordinates, x_interval = np.linspace(200, 400, len(self.visible_nodes_per_layer), retstep=True)
        for layer, x in enumerate(x_coordinates):
            #  node scroll buttons
            if self.visible_nodes_per_layer[layer] < self.network.nodes_per_layer[layer]:
                for text, y_coordinate, adjustment in [('\u2191', 15, -1), ('\u2193', self.winfo_reqheight() - 10, +1)]:
                    button = FormattedButton(self, text=text, width=5,
                                             command=lambda layer_set=layer, adjustment_set=adjustment:
                                             self.change_visible_nodes_row(layer_set, adjustment_set))
                    self.create_window(x, y_coordinate, anchor='center', window=button)

            self.create_text(x, 35, text=f"layer {layer}")
            self.create_text(x, 50, text='Input' if layer == 0 else (
                'Output' if layer == len(self.visible_nodes_per_layer) - 1 else 'Hidden'))
            self.visible_nodes_row_labels.append(self.create_text(x, 65))

            y_coordinates = self.get_node_y_coordinates(layer)
            for node, y in enumerate(y_coordinates):
                node_row = node // self.visible_nodes_per_layer[layer]
                node_state = 'normal' if node_row == self.current_visible_nodes[layer] else 'hidden'
                if layer != len(self.network.nodes_per_layer) - 1:  # draw weights
                    next_y_coordinates = self.get_node_y_coordinates(layer + 1)
                    node_weights = self.network.weights[layer][node]
                    node_weights_colors = 255 * (node_weights - min(node_weights)) / (
                            max(node_weights) - min(node_weights))
                    for next_node, (linked_y, weight) in enumerate(zip(next_y_coordinates, node_weights_colors)):
                        next_node_row = next_node // self.visible_nodes_per_layer[layer + 1]
                        self.create_line(x, y, x + x_interval, linked_y, capstyle=tk.ROUND,
                                         fill='#' + 3 * '{0:02x}'.format(int(weight)),
                                         state=node_state, tags=(f'layer{layer}node_row{node_row}',
                                                                 f'layer{layer + 1}node_row{next_node_row}'))
                    if print_neurons:
                        with np.printoptions(precision=2, suppress=True, threshold=5):
                            print(f"({node}, {layer}), Neuron, weight = {str(self.network.weights[layer][node])}, "
                                  f"bias = {'None' if layer == 0 else f'{self.network.biases[layer - 1][node]:.2f}'}")
                else:
                    if print_neurons:
                        with np.printoptions(precision=2, suppress=True, threshold=5):
                            print(f"({layer}, {node}), Neuron, weight = None, bias = "
                                  f"{self.network.biases[layer - 1][node]:.2f}")
        for layer, x in enumerate(x_coordinates):
            y_coordinates = self.get_node_y_coordinates(layer)
            node_radius = np.clip(200 / self.visible_nodes_per_layer[layer], 0, 25)
            for node, y in enumerate(y_coordinates):
                node_row = node // self.visible_nodes_per_layer[layer]
                node_state = 'normal' if node_row == self.current_visible_nodes[layer] else 'hidden'
                self.nodes.append(self.create_circle(x, y, node_radius, state=node_state, fill='white',
                                                     tags=f'layer{layer}node_row{node_row}'))  # draw nodes
                self.nodes_text.append(self.create_text(x, y, text=node, state=node_state,
                                                        tags=f'layer{layer}node_row{node_row}'))  # label neurons

        self.update_visible_node_layers()

    def create_circle(self, x, y, r, **kwargs):
        return self.create_oval(x - r, y - r, x + r, y + r, **kwargs)


class ImageViewer(tk.Frame):
    """Browse through other images and see what the network predicted from them"""

    def __init__(self, master, filename, network_gui=None):
        super(ImageViewer, self).__init__(master, highlightthickness=0)
        self.configure(background=self.winfo_toplevel()['background'])
        self.network_gui = network_gui
        self.image_label = tk.Label(self, image=None, bg='black', borderwidth=0)
        self.current_image_number = 0

        self.image_data_set = np.loadtxt(filename, delimiter=',', dtype=float)[:, 1:].reshape((-1, 28 ** 2))
        self.current_image_number_string = tk.StringVar()
        self.change_data_set_image(0)
        previous_button = FormattedButton(self, text='\u2190', width=5, command=lambda: self.change_data_set_image(-1))
        current_image_number = tk.Label(self, textvariable=self.current_image_number_string)
        next_button = FormattedButton(self, text='\u2192', width=5, command=lambda: self.change_data_set_image(+1))

        self.image_label.grid(row=0, columnspan=3, sticky='nsew')
        previous_button.grid(row=1, column=0, sticky='nsew')
        current_image_number.grid(row=1, column=1, sticky='nsew')
        next_button.grid(row=1, column=2, sticky='nsew')

        self.grid_rowconfigure(0, weight=1, minsize=280)
        self.grid_columnconfigure(0, weight=1, minsize=70)
        self.grid_columnconfigure(1, weight=1, minsize=140)
        self.grid_columnconfigure(2, weight=1, minsize=70)

    def change_data_set_image(self, adjustment):
        self.current_image_number = (self.current_image_number + adjustment) % len(self.image_data_set)
        self.change_image_from_array(self.image_data_set[self.current_image_number])
        self.current_image_number_string.set(f'#{self.current_image_number}/{len(self.image_data_set)}')

    def change_image_from_array(self, new_image_array: np.array):
        new_image = Image.fromarray(new_image_array.reshape((28, 28)))
        self.change_image(new_image)
        if self.network_gui is not None:
            self.network_gui.display_activations(new_image_array.flatten() / 255 * 0.99 + 0.001)

    def change_image(self, new_image: Image.Image):
        new_image = new_image.resize((280, 280), resample=Image.NEAREST)
        new_tk_image = ImageTk.PhotoImage(new_image, (280, 280))
        self.image_label.config(image=new_tk_image)
        self.image_label.image = new_tk_image


class CustomTrainGUI(tk.Frame):
    """Complete UI that allows the user to inspect ntwork, browse gallery and draw their own images
        Run ui.mainloop() to run after setting up"""

    def __init__(self, master, network, gallery_file, *args, instant_test=True, **kwargs):
        """
        :param master: The tkinter root, can jsut be Tk.tk() to create a new window
        :param network: The network to be displayed
        :param gallery_file: Image files for the network to predict
        :param args: additional network gui parameters
        :param instant_test: If true, when drawing on the user canvas,
                            the image is immediately fed as input into the network
        :param kwargs: additional network gui parameters
        """
        master.configure(background='DarkSeaGreen3')
        super(CustomTrainGUI, self).__init__(master, width=900, height=600)
        self.network = network
        self.network_gui = NetworkGui(self, network, *args, **kwargs)
        image_viewer_title = tk.Label(self, text='Test library images')
        self.user_canvas = UserCanvas(self)
        canvas_title = tk.Label(self, text='Draw images here')
        self.image_viewer = ImageViewer(self, filename=gallery_file, network_gui=self.network_gui)
        self.instant_test = instant_test
        test_button = FormattedButton(self, text='Test', command=self.test, height=0, pady=5)
        self.winfo_toplevel().title('Neural Network Visualiser')

        if self.instant_test:
            self.user_canvas.canvas.bind('<B1-Motion>', lambda event: (self.user_canvas.draw(event), self.test()))
        else:
            test_button.grid(row=4, column=1, sticky='nsew')

        self.network_gui.grid(row=0, column=0, rowspan=5, sticky='nsew')
        image_viewer_title.grid(row=0, column=1, sticky='we')
        self.image_viewer.grid(row=1, column=1, sticky='we')
        canvas_title.grid(row=2, column=1, sticky='we')
        self.user_canvas.grid(row=3, column=1, sticky='we')
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.pack(fill='both', expand=True)

    def test(self):
        user_image = self.user_canvas.get_image()
        if not self.instant_test:
            self.user_canvas.clear_canvas()
        self.image_viewer.change_image(user_image)
        self.image_viewer.current_image_number_string.set('custom')
        network_input = np.asarray(user_image).flatten() / 255 * 0.99 + 0.001
        self.network_gui.display_activations(network_input)


def test_draw_basic_network():
    basic_network = Network([4, 2, 5])
    root = tk.Tk()
    gui = NetworkGui(root, basic_network)
    gui.grid(row=0, column=0, rowspan=5, sticky='nsew')
    root.mainloop()


if __name__ == '__main__':
    numbers_network = Network([784, 75, 10], load_folder='train')
    ui = CustomTrainGUI(tk.Tk(), numbers_network, visible_nodes_per_layer=[28, 25, 10],
                        gallery_file='data_sets/mnist_test.csv', instant_test=False)
    ui.mainloop()
