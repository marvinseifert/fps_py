# Pynoise
This repository contains files for the pynoise package. The package is a collection of functions for generating noise. 
The noise is created using moderngl and shaders, which allows for fast generation of large noise directly on the GPU.
It is possible to display 2 million checkers at 30 fps. 


# Manual
This is how to create and display noise. 

## Installation
Clone the repository, and install the package using poetry:
```bash
poetry install
```

Tested with Python 3.10. You may need `poetry env use python3.10`.

## Running the GUI
To run the GUI, with optional config path, run the following command in the
terminal: </br>
```bash
poetry run fpspy-gui [config-path]
```
This should open the gui window and the noise window. </br>


## CLI
You can also run stimuli directly from the command line:
```bash
poetry run fpspy <path-to-stim>
```

## 3Brain CLI
3Brain setup currently supports the CLI only.
```bash
poetry run fpspy-3brain <path-to-config>
```

## Other scripts
Show information about a stimulus file:
```bash
poetry run fpspy-info <path-to-stim>
```


![GUI](images/gui.PNG)


## Settings
A configuration file is stored in:
```bash
<standard-config-path>/fpspy/settings.toml
```
On Linux and MacOS this is usually:

```bash
~/.config/fpspy/settings.toml
```

On Windows this is usually: </br>
```bash
C:\Users\<username>\AppData\Roaming\fpspy\settings.toml
```

Example settings.toml file:
```toml
# --- Global settings ---
gl_version = [4, 1]
fps = 75

# --- Arduino settings ---
[arduino]
# port = "/dev/ttyUSB0"
port = "dummy"
baud_rate = 9600
trigger_command = "T"


[paths]
# Override if you want to work from non-default directory for stimuli etc.
data_dir = ""

# --- Window definitions ---
[windows."1"]
y_shift = 1000
x_shift = -1080
window_size = [500, 500]
fullscreen = false
style = "transparent"
channels = [0, 1, 2]

[windows."2"]
y_shift = 1000
x_shift = -500
window_size = [500, 500]
fullscreen = false
style = "transparent"
channels = [0, 1, 2]
```

You can change the config path from the GUI, or by passing the config path as
an argument when starting the GUI. If there is no config file, then 
default settings take effect (see src/fpspy/resources/default_config.toml).

### gl_version
This is the version of OpenGL to use. Should be 4, 1</br>

### y_shift and x_shift
When running the program, two windows will be opened. One window is the GUI and the other is the window in which the noise will be displayed. </br>
y_shift and x_shift are used to shift the noise window to the desired position. For example, if the noise shall be displayed on </br>
the secondary monitor, set the x_shift to the width of the primary monitor. Setting the value for y_shift depends on how the monitors</br>
are aligned in Windows, and setting it correctly might need some trial and error. </br>

### window_size
This is the size of the noise window. In this case it is set so that the window fills the entire monitor. </br>

### fullscreen
If fullscreen is set to True, the noise window will be displayed in fullscreen. </br>
**Warning** Fullscreen currently does not work on a secondary monitor. </br>


## Creating noise
The noise can be created using the parameters "checkerboard size", "window size", "noise frequency" and
"noise duration". </br>
**Checkerboard size** refers to the size of a single checker in px. </br>
**Window size** refers to the size of the window  in which the noise will be displayed in px. </br>
**Noise frequency** refers to the frequency by which the checkerboard pattern will be updated </br>
**Noise duration** refers to the duration of the noise in minutes </br>

You can enter the name of the noise file into the field left to the "Generate Noise" button. This file will be stored in /stimuli folder</br>
If you want to have shuffled noise, you can check the "Shuffle" box. The shuffle logic is shuffle every frame </br>
and shuffle 4 positions in x and y, resulting in 16 different positions in total. </br>
The "Estimated size" text shows the estimated size of the noise file. </br>


## Displaying noise

To display noise, select the noise file you want to play in the list and click the "play noise" button. </br>
The noise will be displayed in the window. </br>
To stop the noise, click the "stop noise" button. </br>


## Log file
The log file will be stored in the /logs folder. </br>
At the moment it is very simple, it just logs the time when the noise was started, its parameters and how often it was looped <br/>
## Additional settings
You can loop the noise by increasing the number in the "Loops" field. </br>

Colour logic is experimental and not fully implemented. </br>

## Triggering

If you want to trigger the noise using Arduino open the play_noise.py file and look for the function called </br>
_connect_to_arduino_ and change the port to the port of your Arduino. The script will send a "T" in bytecode to the Arduino</br>
at the specified port every noise frame. </br>

**That's it, enjoy the noise!** </br>
![Noise](images/noise.PNG)

# Future work:
- Implement colour noise (this is already in the shaders, just needs to be updated in the play_noise.py script)
- Implement fullscreen on secondary monitor (this is a bug in moderngl_window) </br>
- Expand so single boxes can be shown and moved around (experimental feature, look at the "moving_box.py" script)
- Better exception handling

