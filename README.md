# fpspy
fpspy is used to display stimuli to screens during biological experiments.

You can:
   - have frames presented at predetermined schedule
   - present to multiple screens
   - configurable mapping between screen and stimulus channels
   - trigger external devices at frame changes (e.g. via Arduino)
   - open stimuli as (or convert to) arrays with shape (frames, height, width, channels).

Presentation is done using OpenGL via the moderngl package, which allows for fast presentation of large stimuli.

fpspy was created to allow very large and long stimuli to be presented with reliability and reproducibility. It was also created so that a small core set of modules could be reused across different setups. Currently, fpspy is being built around two different experimental setups, each having their own GUI and CLI interfaces.

fpspy also has some routines for generating basic visual stimuli, such as checkerboard noise.


## Two presentation setups
The package is being developed while being used on two separate setups:

   1. A single DLP 4500 projector displaying monochrome stimuli, where the light comes from a bank of 6+ LEDs which have their intensity controlled by an Arduino.
   2. Dual DLP 4500 projectors with their images overlain, each connected to 3 separate LEDs, effectively displaying a single 6-channel stimuli. The light intensity is only varied by the projectors themselves, with no Arduino control.


The first setup, while limited to monochrome stimuli, can smoothly present stimuli such as a full-field sinusoidal stimuli. The second setup is limited to the projector mirror arrays, which temporally interleaves colours at a very high frequency to achieve an overall 60 Hz frame rate. Such interleaving mean the setup cannot easily present smoothly varying stimuli. 

The codebase has separate features for each setup, for example, separate `main.py` and `main_3brain.py` entry points (the former being the single projector setup). Despite this bifurcation, there is a shared core set of functionality, primarily the `stim.py` module, where the `StimProgram` and `StimArray` abstractions are defined.




# Manual
This is how to create and display noise. 

## Installation
Clone the repository, and install the package using poetry:

```bash
poetry install
```

Tested with Python 3.10. You may need `poetry env use python3.10`.

## Running programs
You can run entrypoints using poetry, like:

```bash
poetry run <entrypoint-name> [args]
```

where `<entrypoint-name>` is one of the entrypoints defined in `pyproject.toml`. Or, you can source the environment and then run scripts directly with Python:

```bash
eval (poetry env activate)
python ./path/to/script.py [args]
```

The entrypoints defined in `pyproject.toml` map to functions in a module, which allows for multiple entrypoints to be defined in a single module. Running such a module with `python ./path/to/module.py` will simply run the module as a script, and may not run the same function as an entrypoint. 


## Running the GUI
To run the GUI, with optional config path, run the following command in the terminal:

```bash
poetry run fpspy-gui [config-path]
```

This should open the gui window and the presentation windows, one per output screen. From the GUI, you can present stimuli (Play button) and generate a couple of basic noise stimuli (Generate button).


## CLI
You can also run stimuli directly from the command line:
```bash
poetry run fpspy <path-to-stim>
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

On Windows this is usually: 

```bash
C:\Users\<username>\AppData\Roaming\fpspy\settings.toml
```

Example settings.toml file:

```toml
# --- Global settings ---
gl_version = [4, 1]
fps = 75
presentation_delay = 4

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
# RGB values will be interpreted as sRGB by the monitor. 
clear_rgba = [0.73, 0.73, 0.73, 1.0]

[windows."2"]
y_shift = 1000
x_shift = -500
window_size = [500, 500]
fullscreen = false
style = "transparent"
channels = [0, 1, 2]
# RGB values will be interpreted as sRGB by the monitor. 
clear_rgba = [0.73, 0.73, 0.73, 1.0]
```

You can change the config path from the GUI, or by passing the config path as
an argument when starting the GUI. If there is no config file, then 
default settings take effect (see src/fpspy/resources/default_config.toml).

### gl_version
This is the version of OpenGL to use. Should be 4, 1.

### y_shift and x_shift
When running the program, two windows will be opened. One window is the GUI and the other is the window in which the noise will be displayed. y_shift and x_shift are used to shift the noise window to the desired position. For example, if the noise shall be displayed on the secondary monitor, set the x_shift to the width of the primary monitor. Setting the value for y_shift depends on how the monitors are aligned in Windows, and setting it correctly might need some trial and error.

### window_size
This is the size of the noise window. In this case it is set so that the window fills the entire monitor.

### fullscreen
If fullscreen is set to True, the noise window will be displayed in fullscreen.
**Warning** Fullscreen currently does not work on a secondary monitor.


## Log file
The log file will be stored in the /logs folder.
At the moment it is very simple, it just logs the time when the noise was started, its parameters and how often it was looped.

## Additional settings
You can loop the noise by increasing the number in the "Loops" field.

Colour logic is experimental and not fully implemented.

## Triggering
If you want to trigger the noise using Arduino open the play_noise.py file and look for the function called _connect_to_arduino_ and change the port to the port of your Arduino. The script will send a "T" in bytecode to the Arduino at the specified port every noise frame.


## Stimulus types and stimulus files
`main_3brain.py` takes a path to a stimulus file (and an optional config string) as input.
There are two types of input files supported:

   1. a HDF5 file containing a serialized stim.StimArray object
   2. a Python file containing a `to_program(config) -> stim.StimProgram

For many cases, a stimulus can be expressed as a (frames, height, width, channels) array along with some optional additional information: frames to trigger on, fps and zoom. The `stim.StimArray` class encapsulates this information, and `fpspy` can serialize and deserialize such stimuli to/from HDF5 files.
`fpspy` has a shader program called `stim.TextureSequence` that knows how to present a `stim.StimArray` stimulus. 

If it is not practical to store a stimulus as an array, you can create your own stimulus program (see the base protocol `stim.StimProgram`) and put it in a Python file with a `to_program(config) -> stim.StimProgram` function. This function will be called by `fpspy` to get the stimulus program to present.

Create a stim.StimArray object by deserializing it from 
from a stimulus array loaded from a hdf5 file, and from a Python file with a `to_program(config) -> stim.StimProgram` function.


### Exported stimuli
Any stimulus that can be displayed with `fpspy` can also be exported to a stim.StimArray HDF5 file. The exported stimulus will be a serialized `stim.StimArray` object.

```
poetry run fpspy-3brain export <path-to-stim> -o <output-path>
```

## Creating checkerboard noise
The noise can be created using the parameters "checkerboard size", "window size", "noise frequency" and
"noise duration".

 - **Checkerboard size** refers to the size of a single checker in px.
 - **Window size** refers to the size of the window  in which the noise will be displayed in px.
 - **Noise frequency** refers to the frequency by which the checkerboard pattern will be updated
 - **Noise duration** refers to the duration of the noise in minutes

You can enter the name of the noise file into the field left to the "Generate Noise" button. This file will be stored in /stimuli folder. If you want to have shuffled noise, you can check the "Shuffle" box. The shuffle logic is shuffle every frame and shuffle 4 positions in x and y, resulting in 16 different positions in total. The "Estimated size" text shows the estimated size of the noise file. 

## 3Brain specific
It would be nice to have the fpspy package offer a core set of features, and
all of the setup specific stuff to be auxillary. I started moving 3Brain
specific features to the `examples` folder.

Here there is:

    - `examples/gui_cal_3brain.py`: A GUI for presenting stimuli one frame at a time, while interfacing with a Raspberry Pi camera used for taking photos from the projection target.
    - `examples/main_3brain.py`: For presenting stimuli to a real sample.



**That's it, enjoy the noise!** </br>
![Noise](images/noise.PNG)

# Future work:
    - Move the stimulus generation code (checkerboard noise etc) into a separate package, or at least a separate submodule.
    - Implement colour noise (this is already in the shaders, just needs to be updated in the play_noise.py script)
    - Implement fullscreen on secondary monitor (this is a bug in moderngl_window)
    - Expand so single boxes can be shown and moved around (experimental feature, look at the "moving_box.py" script)
    - Better exception handling


# Codebase overview

## `stim.py` connects to `play.py` through the `StimProgram` interface
Understanding `stim.py` and `play.py` (or `play_3brain.py`) is sufficient to understand the core functionality provided by the package. And to get a good overview of the purpose and limitations of both of these, look at the `StimProgram` interface in `stim.py`. The whole codebase bifurcates around this interface: one side designs stimuli, and the other side presents them. If, in some code you write, you create a object that implements `StimProgram`, you can then hand it off to `play.py` or `play_3brain.py` and be confident that it will be presented correctly. 




## StimProgram
A StimProgram offers a render(ctx, frame_idx) method. This will be called by each presenter on each frame. The StimProgram is responsible for setting screen pixel values by making OpenGL calls. There are currently 3 ways to create a StimProgram:

    1. Instantiate an TextureSequence object. This class takes in an array of 
    shape (frames, height, width, channels) and handles rendering each frame as 
    a texture. The TextureSequence is currently the primary way to create 
    stimuli. StimArray and Presentation are used with TextureSequence.
    2. Instantiate a ProceduralShader object. This class takes a fragment shader
    as input, and delegates to it. 
    3. call stim.py::from_script(path) to load a Python module that can create
    a StimProgram. This is the most flexible approach. These python modules 
    can be considered data files in the same sense that an exported stimulus
    array is a data file for a stimulus.


## StimArray
StimArray plugs into a TextureSequence program. It serializes and deserializes the information needed to render a stimulus from a (T, H, W, C) array. For example, on what frames in [0, T) should a trigger be sent on? How many frames per second? Options like "zoom" allow a stimulus to be saved more compactly. Broadcasting of the channel dimension also reduces the array size for monochrome stimuli.


## Script based (incl. shader) stimuli
See `examples/shader_based_stimuli` for some stimuli that are loaded as directories with a Python script entrypoint. The shader approach is suitable for stimuli such as moving bars, where the stimulus cannot be easily compressed as an array.


  

## Codebase, now and in the future
If the project is to be more widely used, it would benefit from being reduced to a small core set of modules. The 3brain specific code has already been moved out into the `examples` folder, such as `examples/gui_cal_3brain.py`. It would be good to move out the other setup specific code, such as `src/fpspy/main.py`, so that the core package is just the reusable components. Furthermore, the stimulus generation code could also be moved out into a separate module or package, and possibly the `arduino.py` module too.

## Some unorganized notes

### Stimulus chaining and looping
It would be nice to be able to run:

```bash
main_3brain.py stimulus1.h5 stimulus2.py stimulus3.h5
```

Each of these stimuli would be invoked by sequential queuing of play commands that would pass the stimulus path to each presenter. It is worth considering whether or not the following would be sufficient:

```bash
main_3brain.py stimulus1.h5
main_3brain.py stimulus2.py
main_3brain.py stimulus3.h5
```
or, using the already supported delay to add a time gap between stimuli:

```bash
main_3brain.py --delay 5 stimulus1.h5
main_3brain.py --delay 5 stimulus2.py
main_3brain.py --delay 5 stimulus3.h5
```


### Channel mask belongs in (is isoloted to) StimArray
The need to repeat a pattern for different LED combinations eventuates using some concept of a channel mask or list. You create one "pattern" and pair it with a list where each element describes the LEDs needed for each separate usage of the pattern. The channel mask is "data" in the same way that the pattern is data. When the StimArray class was being designed (the first StimProgram), it was debated as to whether StimArray would own the channel mask, or whether the channel mask would be given to the presenter separately. We opted to keep it internal to the StimArray. 


#### Previous thoughts leading to this decision
I think that a channel mask should be allowed to be specified in combination with a stimulus array, and that the combination of these two pieces of information should be required to know what was presented. This would imply that the stimulus array alone is not enough to record the stimulus presented. 

This will significantly reduce the storage space required for stimuli, and reduce the save and load times, as many stimuli share the same grayscale data but differ in terms of which LEDs are used.

A half-way solution is to allow a channel mask to be specified in a StimArray; however, this would still mean that the spatial pattern needs to be duplicated an a (T, H, W, 1) array for each channel pattern. This would save having to save (T, H, W, C) arrays for each pattern, which is an improvement, but the fact that there is still duplication suggests that the channel mask does not belong in the StimArray.

A further step worth taking is to accommodate a separation between stimulus data and presentation parameters. Currently, the StimArray alone is enough to know what was presented; however, making it so that StimArray+presentation_params are needed would allow a single (T, H, W, 1) StimArray file to be used for all combinations of channel masks. 

How to allow the specification of the channel masks ahead of time, and saving them so that stimuli can be easily replayed from self-contained files? We may not want a "Presentation" file that has a file pointer to a StimArray file. Instead, the Presentation file should contain all information, including the stimulus data. The solution might be more graceful if it naturally becomes supported by allowing for stimuli lists (play these X stimuli in this order, with these gaps between). Such a feature could have a channel mask. What is the dataum? I guess there would be an array of StimArrays, and a list like [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5] would index into the array of stimuli to specify the order of presentation. Such an index list could be accompanied by a list of channel masks. The channel masks would be a presentation parameter, of which there could be various (such as speed or intensity). I think it would be important to be able to render a Presentation back into a single StimArray, as the StimArray should be able to represent any stimuli. This may be an unrealistic goal, as there are likely presentation options, like lightcrafter options or electrically tunable lens settings, that can't be represented by modifying the stimulus array, and instead are records of how some aspect of the light path is to be modified. Although, it's conceivable that ETL settings may with to be controlled per-frame, with associated trigger values, so really, it's not clear where we will end up with StimArray as being enough to replicate a stimulus presentation.


## Troubleshooting


### Dropped frames with 2 presenters
If one presenter's window is behind the other, the compositor may notices that the behind window is not visible, and it can decide to reduce the framerate dramatically (e.g. to 1 Hz). So, if you are debugging with 2 presenters, make sure neither are occluded.

