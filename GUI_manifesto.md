# GUI
The gui needs to be an instrument panel first. It needs to show what is happening in the fpspy program 
at every moment. But, it should also have a generation tab, which allows the generation of new stimuli. 

### List of instrument functions: 
Arduino control:
- Arduino connection status
- Arduino status if possible look at the Arduino code: [esp32_stimulator](../../CLionProjects/esp32_stimulator)
- Should allow the sending of commands to the connected Ardunio. Some stimuli are hard coded to the Arduino and can be started or stopped by sending a text string to the Arduino. This was present in the previous gui.
- Selection of created stimuli and presentation of the stimulus. (Start, Stop). Change the stimulus folder. 
    - Ideally this is enhanced compared to the previous gui by showing a stimulus preview and allowing to step through a stimulus step by step.
- It also needs a input that sends led information to the Arduino for a specific stimuli. For example, BW Noise can be displayed using white leds or a single one (led_610)

### Organizational functions

- The user should be able to create new noise stimuli using the gui
- Possibly also new moving bar stimuli
- This should be open to whatever new stimulus templates are created. 

### Log functions
Ideally there is a third tab which allows the user to look at the logs of previous stimuli to check if frames were dropped.

### Design principals
look at gui_design_principles.txt