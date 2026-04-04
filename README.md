# YouTube Imitation Learning Game Bot
Train a game bot on YouTube videos.

## Concept
Train an inverse dynamics model (IDM) which predicts the actions (keystrokes and mouse movent) taken from pairs of before and after game images (screenshots). Use the trained IDM to label YouTube videos. Train a policy on the labelled YouTube videos.

## Philosophy
Keep the policy model lightweight since it has to play in realtime. The IDM can be as heavy as desired since it is only used to label data.

## Setup

- change all instances of "EternalJK" to your game window name
- launch your game and join a multiplayer server
- run ```python3 jka_2026_data_recorder.py```
- pip install any necessary libraries if it errors out
- once its running, press R to record
- use the actions WASD, space, ctrl, left mouse click, right mouse click and mouse movement to record a short demo of gameplay
- press R to stop recording
- press P to play it back
- confirm that the playback matches the recording
- you're now ready to record data to train the IDM!
- record a decent amount of gameplay actions in different locations on the map and different actions
- note the name of the saved recording
- press Q to exit recorder program

- you now have adequate data to train your IDM
- change the name of the expected recording in jka_2026_train_idm.py
- run ```python3 jka_2026_train_idm.py```
- you now have a trained IDM!
- now there are 2 options:

### Train On A Single YouTube Video

- run ```python3 jka_2026_label_youtube_with_idm.py```, this will label your YouTube video with actions predicted by the IDM
- run ```python3 jka_2026_train_policy_on_youtube.py```, this will train your policy network on the labelled YouTube data
- run ```python3 jka_2026_controller.py```, this will let the trained policy network play the game (press C to stop)

### Train On Many YouTube Videos

- run ```python3 jka_2026_youtube_pipeline.py``` after picking a YouTube channel and changing it in the file
- this will train on all YouTube videos belonging to that channel
- run ```python3 jka_2026_controller.py```, this will let the trained policy network play the game (press C to stop)

## Credits

Some of the interfacing code and much of the inspiration is borrowed from https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning. The key difference is that they have CSGO specific code that reads game memory to get values such as score. Our approach can be applied easily to any game. Another difference is they use an LSTM whereas we keep it simple, feedforward. This, along with some other optimizations, allows us to run the bot at 60 fps on a modest 8GB GPU. Our policy network is adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py.
